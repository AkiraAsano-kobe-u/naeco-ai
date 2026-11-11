"""3〜4 席モデル用の観測・行動エンコーダ契約(Phase 10 D2 / core-api §8.3)。

既存 `naeco/train/encoding.py` は 2 席 gen1300 系の hash 対象なので byte 単位で
変更しない。多人数モデルの契約はこのモジュールに隔離する:

- 観測: OBS_ENCODER_VERSION=2.0.0。特徴の算出は旧 encoder(人数一般で書かれた
  `encoding.encode_observation`)へ委譲し、入力 688(3 席)/807(4 席)。
  n=2 の観測は受け付けない(2 席は旧契約だけが扱う)。
- 行動: 意味を変えないので ACTION_ENCODER_VERSION=1.0.0、hash も旧定義のまま。
- 新 obs hash は本モジュールのソースと、委譲先である旧 encoder の obs hash の
  両方を含む(どちらが変わっても多人数 checkpoint は拒否される)。

`contract_for(num_players)` が人数から契約を選ぶ唯一の入口で、checkpoint・
agent・search_evaluator はここを通す。呼び出し側が版や hash を指定する経路は
無い。標準ライブラリのみを使う(torch 境界は net.py / ppo.py)。
"""
import collections
import hashlib
import os

from naeco.train import encoding as legacy

OBS_ENCODER_VERSION = "2.0.0"
ACTION_ENCODER_VERSION = legacy.ACTION_ENCODER_VERSION
SUPPORTED_PLAYERS = (3, 4)
ALL_PLAYERS = (2,) + SUPPORTED_PLAYERS


def _require_multi(num_players):
    if type(num_players) is not int or num_players not in SUPPORTED_PLAYERS:
        raise ValueError(
            "multi-seat encoder supports num_players in %r, got %r"
            % (SUPPORTED_PLAYERS, num_players))
    return num_players


def obs_dim(num_players):
    """多人数観測の次元(3 席 688 / 4 席 807)。"""
    return legacy.obs_dim(_require_multi(num_players))


def action_dim():
    return legacy.action_dim()


def encode_observation(obs, cards=None):
    """3/4 席の観測を float list へ変換する。n=2 は旧契約の領分なので拒否する。"""
    _require_multi(obs.num_players)
    return legacy.encode_observation(obs, cards)


def encode_action(obs, action, cards=None):
    _require_multi(obs.num_players)
    return legacy.encode_action(obs, action, cards)


def _source_bytes():
    with open(os.path.abspath(__file__), "rb") as fh:
        return fh.read()


def encoder_sha256():
    """(obs_encoder_sha256, action_encoder_sha256) を返す。

    obs は本モジュールのソースと旧 encoder の obs hash を連結した上で hash する。
    action は旧定義の hash をそのまま返す(行動特徴の意味は変えない)。
    """
    legacy_obs, legacy_action = legacy.encoder_sha256()
    payload = (_source_bytes() + b"|legacy-obs:" + legacy_obs.encode("ascii")
               + b"|version:" + OBS_ENCODER_VERSION.encode("ascii"))
    return (hashlib.sha256(payload + b"|obs-multi").hexdigest(), legacy_action)


EncoderContract = collections.namedtuple(
    "EncoderContract",
    ("num_players", "obs_encoder_version", "action_encoder_version",
     "encoder_sha256", "encode_observation", "encode_action", "obs_dim",
     "action_dim"))


def contract_for(num_players):
    """人数から encoder 契約を選ぶ(Phase 10 D2)。2..4 以外・bool は ValueError。

    n=2 は旧 `encoding` モジュールそのもの(関数も hash も既存のまま)。
    """
    if type(num_players) is not int or num_players not in ALL_PLAYERS:
        raise ValueError("num_players must be an integer in 2..4, got %r"
                         % (num_players,))
    if num_players == 2:
        return EncoderContract(
            2, legacy.OBS_ENCODER_VERSION, legacy.ACTION_ENCODER_VERSION,
            legacy.encoder_sha256, legacy.encode_observation,
            legacy.encode_action, legacy.obs_dim(2), legacy.action_dim())
    return EncoderContract(
        num_players, OBS_ENCODER_VERSION, ACTION_ENCODER_VERSION,
        encoder_sha256, encode_observation, encode_action,
        obs_dim(num_players), action_dim())


def _net_dims(net):
    arch = getattr(net, "arch_dict", None)
    if arch is None:
        return None
    arch = arch()
    return arch.get("obs_dim"), arch.get("action_dim")


def encode_for_net(net, observation, with_actions=True):
    """観測の人数で契約を選んでエンコードし、net の入力次元と照合する。

    戻り値は (obs_vec, action_vecs)。with_actions=False なら action_vecs は None
    (非手番席の value 推論用)。net が `arch_dict` を持てば obs_dim / action_dim
    を照合し、別人数のモデルへ観測を流す誤用を ValueError で止める(フェイク net
    は照合しない)。
    """
    contract = contract_for(observation.num_players)
    obs_vec = contract.encode_observation(observation)
    action_vecs = None
    if with_actions:
        action_vecs = [contract.encode_action(observation, act)
                       for act in observation.legal_actions]
    dims = _net_dims(net)
    if dims is not None:
        want_obs, want_action = dims
        if want_obs is not None and want_obs != len(obs_vec):
            raise ValueError(
                "net expects obs_dim %r but a %d-seat observation encodes to %d"
                % (want_obs, observation.num_players, len(obs_vec)))
        if (want_action is not None and action_vecs
                and want_action != len(action_vecs[0])):
            raise ValueError("net expects action_dim %r, got %d"
                             % (want_action, len(action_vecs[0])))
    return obs_vec, action_vecs
