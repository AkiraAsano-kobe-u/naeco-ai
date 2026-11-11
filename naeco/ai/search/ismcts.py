"""自己対戦用 IS-MCTS による配置手探索。"""
import collections
import hashlib
import math
import random
import time

from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.sampler import sample_hidden_state
from naeco.engine import game
from naeco.engine.cards import load_cards
from naeco.engine.observe import observe, serialize_observation
from naeco.engine.state import Phase, derive_seed

from .common import (Budget, H, SearchStats, choose_heuristic_action,
                     evaluate_static, merge_deadline_overruns, rollout)


def evaluate_with_evaluator(evaluator, state, cards, stats):
    """決定化済み state の葉評価を評価器で置き換える(D-6)。

    `evaluate_static` と同じ位置・同じ戻り値契約([0,1] のプレイヤーベクトル)
    で呼ぶ。終端(`game.actor` が None)は `terminal_utilities` の ±1 を (u+1)/2
    で [0,1] へ写像する。非終端は
    `evaluator.evaluate(observe(state, actor, cards))` を呼び、返った
    [0,1] スカラー p から 2人ゼロ和のプレイヤーベクトル [p, 1-p](絶対席順)を
    組み立てる(D-2)。評価器へ state は渡さず、観測だけを渡す(D-6)。
    呼出数は既存の `stats.evaluations` へ計上するが、評価器を**実際に呼ぶ**
    直前だけ1増やす(終端・合法手ゼロの退化分岐は forward しないので計上
    しない。Phase 5 の net 呼出数の実測はこの計上を正本とする)。

    3/4 席(Phase 10 D3)は `_evaluate_multi` へ分岐する。n=2 の分岐・計算順は
    上記のまま変えない。
    """
    if state.num_players != 2:
        return _evaluate_multi(evaluator, state, cards, stats)
    actor = game.actor(state)
    if actor is None:
        utilities = game.terminal_utilities(state, cards)
        return [(u + 1) / 2 for u in utilities]
    observation = observe(state, actor, cards)
    if not observation.legal_actions:
        # 合法手ゼロの退化状態(予算打切りの探索途中で到達し得る防御分岐)。
        # 評価器の契約は legal_actions 非空なので、中立のゼロ和値を返す。
        return [0.5, 0.5]
    stats.evaluations += 1
    unused_priors, p = evaluator.evaluate(observation)
    return [p if pid == actor else 1.0 - p
            for pid in range(state.num_players)]


def _evaluate_multi(evaluator, state, cards, stats):
    """3/4 席の葉評価: 各席の自席スカラーを絶対席順の n ベクトルへ並べる(D3)。

    - 終端: D1 の順位線形効用 u_i を (u_i+1)/2 へ写す。
    - 手番席に合法手が無い退化状態: 中立の n 個の 0.5(n=2 の [0.5,0.5] と同じ意味)。
    - それ以外: 席 pid = 0..n-1 について `observe(state, pid)` だけを
      `evaluator.evaluate_value` へ渡す(真の hidden state は渡さない)。評価器は
      u を clip 後 (u+1)/2 にした [0,1] を返す。相手の残り価値を均等配分しない。
      推論回数は各席の実呼出ごとに `stats.evaluations` を 1 増やす。
    """
    n = state.num_players
    actor = game.actor(state)
    if actor is None:
        utilities = game.terminal_utilities(state, cards)
        return [(u + 1) / 2 for u in utilities]
    if not game.legal_actions(state, cards):
        return [0.5] * n
    values = []
    for pid in range(n):
        observation = observe(state, pid, cards)
        stats.evaluations += 1
        values.append(float(evaluator.evaluate_value(observation)))
    return values


def _rollout_with_evaluator(state, budget, cards, heuristic, horizon,
                            start_round, stats, evaluator):
    """common.rollout と同じ構造で、葉評価だけを評価器へ置き換える。"""
    limit = start_round + horizon
    while state.phase != Phase.GAME_OVER and state.round_number <= limit:
        if budget.check() == 0:
            stats.truncated_samples += 1
            return evaluate_with_evaluator(evaluator, state, cards, stats)
        action = choose_heuristic_action(state, cards, heuristic)
        if action is None:
            break
        state, unused_used = budget.step(state, action, cards)
    return evaluate_with_evaluator(evaluator, state, cards, stats)


class _Node(object):
    """情報集合木のノード。行動辞書は Action をキーに保つ。"""

    def __init__(self, actor):
        self.actor = actor
        self.visits = 0
        self.availability = collections.defaultdict(int)
        self.edge_visits = collections.defaultdict(int)
        self.value_sum = {}
        self.children = {}
        # S2: policy prior(action -> prob)。ルートは探索前の事前計算(act 内)、
        # 子ノードは展開時に評価器が返したものを保存する。evaluator=None の
        # 後方互換では空のまま(選択は従来 UCB)。
        self.priors = {}


class IsmctsAgent(object):
    """設計書 §6.7 の player-vector SO-ISMCTS。"""

    def __init__(self, root_seed, budget_transitions, c=1.41, c_puct=1.5,
                 horizon=H, cards=None, budget_seconds=None,
                 clock=time.monotonic, evaluator=None, policy_agent=None,
                 root_policy="puct", gumbel_m=8, c_visit=50.0, c_scale=1.0,
                 gumbel_scale=1.0, opponent_mixture=None):
        """evaluator: 葉評価を置き換える Evaluator(D-6)。None なら従来の静的評価。

        root_policy: "puct"(既定。Phase 5 認定構成)か "gumbel"(Phase 7 段2)。
        "gumbel" は根で Gumbel top-m の非復元抽出 + sequential halving を行い、
        completed Q から改善方策 π'(a) ∝ exp(log π(a) + σ(Q̂(a))) を
        `last_root` へ記録する(Danihelka ら 2022)。評価器が必須。
        gumbel_m: 根の候補数の上限。c_visit / c_scale: σ(q) = (c_visit + max_N) c_scale q。
        gumbel_scale: Gumbel ノイズの尺度(0 で決定論的 top-m)。

        c_puct: PUCT の探索係数(S2)。evaluator 使用時の選択にだけ使う
        (UCB の c とは別記号・別値で、流用しない)。既定 1.5。
        evaluator=None では使われず、従来の UCB(c)のまま(後方互換)。

        policy_agent: pending / 一択局面の委譲先(D-1)。None なら従来どおり
        HeuristicAgent へ委譲する(後方互換)。

        opponent_mixture: Phase 12b 手法 C(D2)。`PolicyMixture` を渡すと、木の
        降下で**根の手番席以外**の選択だけを、探索ごとに 1 つ抽選して固定した
        成分方策からの合法手サンプルに置き換える。根の手番席は従来どおり
        PUCT/UCB、葉評価と backup は同じ n 成分ベクトル。成分方策に渡すのは
        その決定化 state の手番席の観測(`observe(state, actor)`)だけで、
        真の hidden state・実対戦の相手種別は渡さない。n=2 の観測では拒否する。
        None(既定)の経路は従来と完全に同じ。
        """
        if opponent_mixture is not None:
            from .evaluator import PolicyMixture
            if not isinstance(opponent_mixture, PolicyMixture):
                raise ValueError("opponent_mixture must be a PolicyMixture")
        self._mixture = opponent_mixture
        self._active_component = None
        self.last_mixture_id = None
        self.opponent_policy_calls = 0
        self._root_seed = root_seed
        self._budget_transitions = budget_transitions
        self._budget_seconds = budget_seconds
        self._clock = clock
        self._c = c
        self._c_puct = c_puct
        self._horizon = horizon
        self._cards = cards or load_cards()
        self._heuristic = HeuristicAgent(0)
        self._evaluator = evaluator
        self._policy_agent = policy_agent
        if root_policy not in ("puct", "gumbel"):
            raise ValueError("root_policy must be 'puct' or 'gumbel'")
        if root_policy == "gumbel" and evaluator is None:
            raise ValueError("root_policy='gumbel' requires an evaluator")
        if gumbel_m < 1:
            raise ValueError("gumbel_m must be >= 1")
        self._root_policy = root_policy
        self._gumbel_m = int(gumbel_m)
        self._c_visit = float(c_visit)
        self._c_scale = float(c_scale)
        self._gumbel_scale = float(gumbel_scale)
        self.last_root = None
        self.stats = SearchStats()

    def _decision_seed(self, observation):
        digest = hashlib.sha256(serialize_observation(observation)).digest()[:8]
        return derive_seed(self._root_seed, "search", int.from_bytes(digest, "big"))

    def _ucb(self, node, action, actor):
        """従来の UCB1(evaluator=None の後方互換で使う)。

        actor は**現在の決定化 state の手番**(`game.actor(state)`)。node.actor
        は展開時の決定化で決まった値であり、別の決定化では同じ経路が終端
        (actor=None)や別の手番に到達しうる(Phase 7 段1 で実クラッシュ)。
        """
        visits = node.edge_visits[action]
        if visits == 0:
            return float("inf")
        return (node.value_sum[action][actor] / visits
                + self._c * math.sqrt(math.log(node.availability[action]) / visits))

    def _puct(self, node, action, actor):
        """PUCT: policy prior を選択バイアスへ使う(S2)。actor は現在の決定化
        state の手番(`_ucb` と同じ理由で node.actor を使わない)。

        PUCT(a) = Q(a) + c_puct * prior(a) * sqrt(availability[a]) / (1 + edge_visits[a])
        Q(a) は edge_visits[a]==0 なら FPU=0.5(中立)、それ以外は
        value_sum[a][actor] / edge_visits[a]。未訪問は +inf にせず全合法手を
        同一尺度で比べる(edge_visits=0 では prior 項が最大になり、prior の
        高い未訪問手から優先的に展開される)。prior が無い action は
        prior=0 として扱う(防御分岐)。
        """
        visits = node.edge_visits[action]
        if visits == 0:
            q = 0.5
        else:
            q = node.value_sum[action][actor] / visits
        prior = node.priors.get(action, 0.0)
        return q + self._c_puct * prior * math.sqrt(node.availability[action]) / (1.0 + visits)

    def _resolve_pending(self, state, budget):
        """木で扱わない下位決定をヒューリスティックへ委譲する。"""
        used = 0
        while state.pending is not None and state.phase != Phase.GAME_OVER:
            if budget.remaining == 0:
                return state, used, True
            action = choose_heuristic_action(state, self._cards, self._heuristic)
            if action is None:
                return state, used, False
            state, one = budget.step(state, action, self._cards)
            used += one
        return state, used, False

    def _simulate(self, root, observation, decision_seed, iteration, budget,
                  forced=None):
        """1反復(決定化 → 木降下 → 葉評価/rollout → backup)。

        戻り値は (根で選んだ行動 or None, 消費した遷移数)。None は根行動を選ぶ
        前に予算/期限が尽きた場合。`forced` を与えると根ではその行動を選ぶ
        (Gumbel 根選択用)。
        """
        before = budget.spent
        state = sample_hidden_state(
            observation, derive_seed(decision_seed, "determinize", iteration),
            self._cards)
        root_round = state.round_number
        self.stats.sampler_calls += 1
        node = root
        path = []
        truncated = False
        opponent_rng = None
        while True:
            state, unused_lower, truncated = self._resolve_pending(state, budget)
            # `_resolve_pending` は「最後の1遷移で pending を解決し終えた」場合に
            # truncated=False を返すが、そのとき残予算は 0 になっている。
            # ここで確認しないと直後の budget.step が例外を投げる。
            if truncated or budget.check() == 0 or state.phase == Phase.GAME_OVER:
                break
            legal = sorted(game.legal_actions(state, self._cards),
                           key=lambda action: action.sort_key())
            if not legal:
                break
            acting = game.actor(state)
            if acting is None:
                raise AssertionError(
                    "non-terminal state without actor in ISMCTS selection")
            for action in legal:
                node.availability[action] += 1
            if forced is not None and node is root:
                # Gumbel 根選択(段2): 根では sequential halving が決めた
                # 候補を強制し、根より下は従来どおり PUCT で選ぶ。
                action = forced
                expand = action not in node.children
            elif self._mixture is not None and acting != observation.player_id:
                # 手法 C: 他家の選択は探索ごとに固定した成分方策からサンプル。
                if opponent_rng is None:
                    opponent_rng = random.Random(
                        derive_seed(decision_seed, "opponent", iteration))
                action = self._sample_opponent(state, acting, opponent_rng)
                expand = action not in node.children
            elif self._evaluator is None:
                # 後方互換: 従来どおり unexpanded[0] 強制展開 + UCB 選択。
                unexpanded = [action for action in legal
                              if action not in node.children]
                if unexpanded:
                    action = unexpanded[0]
                    expand = True
                else:
                    action = sorted(
                        legal,
                        key=lambda candidate: (-self._ucb(node, candidate, acting),
                                                candidate.sort_key()))[0]
                    expand = False
            else:
                # S2: 未展開も含めて全合法手を PUCT スコアで選択する
                # (unexpanded[0] 強制展開は使わない。prior が展開順に効く)。
                action = sorted(
                    legal,
                    key=lambda candidate: (-self._puct(node, candidate, acting),
                                            candidate.sort_key()))[0]
                expand = action not in node.children
            if expand:
                state, unused_one = budget.step(state, action, self._cards)
                path.append((node, action))
                state, unused_lower, truncated = self._resolve_pending(state, budget)
                child_actor = game.actor(state)
                child = _Node(child_actor)
                node.children[action] = child
                node.value_sum[action] = [0.0] * observation.num_players
                if self._evaluator is not None and child_actor is not None:
                    # S2: 決定化済み state の観測で評価器を呼び、policy
                    # prior を子ノードへ保存する。展開時の value は選択の
                    # seed に使わない(葉評価は従来どおり rollout 葉の value)。
                    child_obs = observe(state, child_actor, self._cards)
                    if child_obs.legal_actions:
                        priors, unused_expansion_value = (
                            self._evaluator.evaluate(child_obs))
                        child.priors = dict(priors)
                        self.stats.evaluations += 1
                self.stats.nodes_created += 1
                self.stats.total_nodes += 1
                node = child
                break
            state, unused_one = budget.step(state, action, self._cards)
            path.append((node, action))
            node = node.children[action]
            if budget.remaining == 0:
                truncated = True
                break
        if budget.check() == 0:
            truncated = True
            self.stats.truncated_samples += 1
            if self._evaluator is None:
                value = evaluate_static(state, self._cards, self.stats)
            else:
                value = evaluate_with_evaluator(
                    self._evaluator, state, self._cards, self.stats)
        else:
            if self._evaluator is None:
                value = rollout(state, budget, self._cards,
                                self._heuristic, self._horizon,
                                root_round, self.stats)
            else:
                value = _rollout_with_evaluator(
                    state, budget, self._cards, self._heuristic,
                    self._horizon, root_round, self.stats,
                    self._evaluator)
        for current, action in path:
            current.visits += 1
            current.edge_visits[action] += 1
            for player in range(observation.num_players):
                current.value_sum[action][player] += value[player]
        return (path[0][1] if path else None), budget.spent - before

    def _sample_opponent(self, state, acting, rng):
        """手法 C: 決定化 state の手番席観測だけを成分方策へ渡して 1 手サンプル。"""
        from .evaluator import checked_probabilities
        observation = observe(state, acting, self._cards)
        policy = self._active_component[1]
        probs = checked_probabilities(policy, observation)
        self.opponent_policy_calls += 1
        legal = list(observation.legal_actions)
        return rng.choices(legal, weights=probs, k=1)[0]

    # ------------------------------------------------------------------ Gumbel
    def _gumbel_noise(self, decision_seed, actions):
        rng = random.Random(derive_seed(decision_seed, "gumbel", 0))
        noise = {}
        for action in actions:
            u = rng.random()
            while u <= 0.0:
                u = rng.random()
            noise[action] = self._gumbel_scale * (-math.log(-math.log(u)))
        return noise

    def _completed_q(self, root, actions, actor, root_value):
        """completed Q(未訪問は v_mix で補完)と max_N を返す。"""
        q = {}
        for action in actions:
            visits = root.edge_visits[action]
            if visits > 0:
                q[action] = root.value_sum[action][actor] / visits
        total_visits = sum(root.edge_visits[action] for action in actions)
        if q:
            weight = sum(root.priors.get(action, 0.0) for action in q)
            if weight > 0.0:
                mixed = sum(root.priors.get(action, 0.0) * q[action]
                            for action in q) / weight
            else:
                mixed = sum(q.values()) / len(q)
            v_mix = (root_value + total_visits * mixed) / (1.0 + total_visits)
        else:
            v_mix = root_value
        completed = dict((action, q.get(action, v_mix)) for action in actions)
        max_visits = max([root.edge_visits[action] for action in actions] or [0])
        return completed, max_visits

    def _sigma(self, q, max_visits):
        return (self._c_visit + max_visits) * self._c_scale * q

    def _search_gumbel(self, root, observation, actions, decision_seed, budget,
                       samples, transitions, root_value):
        """根: Gumbel top-m + sequential halving。改善方策を `last_root` に記録する。"""
        actor = root.actor
        logits = dict((action, math.log(max(root.priors.get(action, 0.0), 1e-12)))
                      for action in actions)
        noise = self._gumbel_noise(decision_seed, actions)
        m = min(self._gumbel_m, len(actions))
        candidates = sorted(
            actions, key=lambda a: (-(noise[a] + logits[a]), a.sort_key()))[:m]
        phases = max(1, int(math.ceil(math.log(m, 2)))) if m > 1 else 1
        total = budget.transitions
        iteration = 0
        stop = False
        for phase in range(phases):
            if phase == phases - 1:
                target = total
            else:
                target = int(round(total * (phase + 1) / phases))
            while not stop and budget.check() and budget.spent < target:
                for action in candidates:
                    if not budget.check() or budget.spent >= target:
                        break
                    root_action, spent = self._simulate(
                        root, observation, decision_seed, iteration, budget,
                        forced=action)
                    iteration += 1
                    if root_action is not None:
                        samples[root_action] += 1
                        transitions[root_action] += spent
                    elif budget.deadline_expired:
                        stop = True
                        break
                    else:
                        raise AssertionError(
                            "ISMCTS iteration did not select a root action")
            if stop:
                break
            if phase < phases - 1 and len(candidates) > 1:
                completed, max_visits = self._completed_q(
                    root, actions, actor, root_value)
                candidates = sorted(
                    candidates,
                    key=lambda a: (-(noise[a] + logits[a]
                                     + self._sigma(completed[a], max_visits)),
                                   a.sort_key()))[:max(1, len(candidates) // 2)]
        completed, max_visits = self._completed_q(root, actions, actor, root_value)
        chosen = sorted(
            candidates,
            key=lambda a: (-(noise[a] + logits[a]
                             + self._sigma(completed[a], max_visits)),
                           a.sort_key()))[0]
        self._record_root(root, actions, actor, root_value, chosen, m)
        return chosen, iteration

    def _record_root(self, root, actions, actor, root_value, chosen, candidates):
        """根の統計と改善方策 π'(a) ∝ exp(log π(a) + σ(Q̂(a))) を `last_root` へ残す。

        PUCT 経路(既定)でも評価器があれば記録する。学習ターゲットの選択
        (訪問分布か改善方策か)は学習側の判断に委ね、ここでは両方を残す。
        """
        completed, max_visits = self._completed_q(root, actions, actor, root_value)
        logits = [math.log(max(root.priors.get(a, 0.0), 1e-12)) for a in actions]
        improved_logits = [lg + self._sigma(completed[a], max_visits)
                           for lg, a in zip(logits, actions)]
        peak = max(improved_logits)
        weights = [math.exp(x - peak) for x in improved_logits]
        norm = sum(weights)
        self.last_root = {
            "action_list": list(actions),
            "actions": [list(a.sort_key()) for a in actions],
            "prior": [root.priors.get(a, 0.0) for a in actions],
            "visits": [root.edge_visits[a] for a in actions],
            "completed_q": [completed[a] for a in actions],
            "improved_policy": [w / norm for w in weights],
            "root_value": root_value,
            "candidates": candidates,
            "chosen_index": actions.index(chosen),
        }

    def act(self, observation):
        legal_root = observation.legal_actions
        if not legal_root:
            raise ValueError("no legal actions for this observer")
        if observation.pending is not None or len(legal_root) == 1:
            if self._policy_agent is not None:
                # 委譲先 policy_agent は1決定で1回 net.infer() を呼ぶため、
                # net 呼出総数 = evaluations + delegated_calls が成り立つ。
                self.stats.delegated_calls += 1
                return self._policy_agent.act(observation)
            return self._heuristic.act(observation)
        if self._budget_transitions < 1:
            raise ValueError("budget_transitions must be positive")

        actions = sorted(legal_root, key=lambda action: action.sort_key())
        root = _Node(observation.player_id)
        if self._evaluator is not None:
            # S2: 探索前にルート prior を事前計算する。最初の selection から
            # 全合法手に prior が存在する。評価器は観測だけを受け取る(D-6)。
            root_priors, root_value = self._evaluator.evaluate(observation)
            root.priors = dict(root_priors)
            self.stats.evaluations += 1
        else:
            root_value = None
        decision_seed = self._decision_seed(observation)
        if self._mixture is not None:
            if observation.num_players < 3:
                raise ValueError(
                    "opponent policy mixture search (method C) is for 3/4 seats")
            self._active_component = self._mixture.sample_component(
                derive_seed(decision_seed, "mixture", 0))
            self.last_mixture_id = self._active_component[0]
        budget = Budget(self._budget_transitions,
                        seconds=self._budget_seconds,
                        clock=self._clock).start()
        samples = dict((action, 0) for action in actions)
        transitions = dict((action, 0) for action in actions)
        iteration = 0
        self.stats.nodes_created += 1
        self.stats.total_nodes += 1
        if self._root_policy == "gumbel":
            chosen, iteration = self._search_gumbel(
                root, observation, actions, decision_seed, budget,
                samples, transitions, root_value)
        else:
            while budget.check():
                root_action, spent = self._simulate(
                    root, observation, decision_seed, iteration, budget)
                iteration += 1
                if root_action is not None:
                    samples[root_action] += 1
                    transitions[root_action] += spent
                elif budget.deadline_expired:
                    # サンプラー中に期限を跨いだ場合は、根行動を選ぶ前に切れた正常な
                    # 早期停止として扱う。0標本の監査は下の gap_counts で明示される。
                    break
                else:
                    raise AssertionError(
                        "ISMCTS iteration did not select a root action")
            chosen = sorted(actions, key=lambda action: (-root.edge_visits[action],
                                                          action.sort_key()))[0]
            if self._evaluator is not None:
                self._record_root(root, actions, root.actor, root_value, chosen,
                                  len(actions))

        gap_counts = [samples[action] for action in actions]
        gap = max(gap_counts) - min(gap_counts)
        zeros = sum(1 for count in gap_counts if count == 0)
        self.stats.max_sample_gap.append(gap)
        self.stats.zero_sample_actions.append(zeros)
        if zeros:
            self.stats.decisions_with_zero_sample_actions += 1
        self.stats.decisions += 1
        self.stats.transitions += budget.spent
        self.stats.determinizations += iteration
        decision_ms = budget.elapsed_ms
        self.stats.elapsed_ms.append(decision_ms)
        self.stats.search_elapsed_ms += decision_ms
        merge_deadline_overruns(budget, self.stats)
        self.stats.samples_per_action.append(
            dict((str(action.sort_key()), samples[action]) for action in actions))
        self.stats.transitions_per_action.append(
            dict((str(action.sort_key()), transitions[action]) for action in actions))
        return chosen
