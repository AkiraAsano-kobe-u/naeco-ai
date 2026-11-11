"""単純決定化 (PIMC) による配置手探索。"""
import hashlib
import time
from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.sampler import sample_hidden_state
from naeco.engine.cards import load_cards
from naeco.engine.observe import serialize_observation
from naeco.engine.state import derive_seed

from .common import (Budget, H, SearchStats, merge_deadline_overruns,
                     rollout_from)


class PimcAgent(object):
    """設計書 §6.6 の局所予算 PIMC。"""

    def __init__(self, root_seed, budget_transitions, horizon=H, cards=None,
                 budget_seconds=None, clock=time.monotonic):
        self._root_seed = root_seed
        self._budget_transitions = budget_transitions
        self._budget_seconds = budget_seconds
        self._clock = clock
        self._horizon = horizon
        self._cards = cards or load_cards()
        self._heuristic = HeuristicAgent(0)
        self.stats = SearchStats()

    def act(self, observation):
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        if observation.pending is not None or len(legal) == 1:
            return self._heuristic.act(observation)
        actions = sorted(legal, key=lambda action: action.sort_key())
        total_budget = self._budget_transitions
        if total_budget < len(actions):
            raise ValueError("budget must allow at least one sample per action")

        digest = serialize_observation(observation)
        decision_seed = derive_seed(
            self._root_seed, "search", int.from_bytes(
                hashlib.sha256(digest).digest()[:8], "big"))
        budget = Budget(total_budget, seconds=self._budget_seconds,
                        clock=self._clock).start()
        totals = dict((action, 0.0) for action in actions)
        counts = dict((action, 0) for action in actions)
        transitions = dict((action, 0) for action in actions)

        # 決定化の個数 K は**予算から決まる**(固定値にしない)。1つの決定化で
        # 全行動を1回ずつ評価し、予算が残っていれば次の決定化へ進む。
        #
        # 枠ごとに局所予算を割り当てて「使い切るまで標本を繰り返す」方式は誤りだった:
        # ロールアウト方策が決定論的なので、同じ (決定化, 行動) の2標本目以降は
        # **まったく同じ結果**になり、予算が情報ゼロの再計算に消える。さらに局所予算が
        # 1標本ぶん(実測 28〜32遷移)に満たないと、全標本が途中打切りになる。
        passes = 0
        while budget.check():
            determinization = sample_hidden_state(
                observation, derive_seed(decision_seed, "determinize", passes),
                self._cards)
            self.stats.sampler_calls += 1
            # 巡回の先頭を毎回ずらし、端数の恩恵が先頭の行動へ偏らないようにする。
            offset = passes % len(actions)
            order = actions[offset:] + actions[:offset]
            for position, action in enumerate(order):
                # 各 sample/action 境界で期限を確認する。初巡も壁時計では例外にしない。
                left = budget.check()
                if left <= 0:
                    break
                # 「まだ1標本も取っていない行動」のぶんを最低1遷移ずつ残す。
                # 残さないと最初の標本が予算を食い尽くし、後続の行動が0標本になる。
                reserve = (len(order) - 1 - position) if passes == 0 else 0
                cap = left - reserve
                sub = budget.cap(cap)
                value, used = rollout_from(
                    determinization, action, sub, self._cards,
                    self._heuristic, self._horizon, self.stats)
                if used < 1:
                    raise AssertionError("PIMC rollout did not advance")
                budget.transitions -= used
                budget.spent += used
                totals[action] += value[observation.player_id]
                counts[action] += 1
                transitions[action] += used
            passes += 1

        sampled = [action for action in actions if counts[action] > 0]
        if not sampled:
            # 期限が最初の境界で切れた壁時計モード。0標本を明示して決定論的な
            # fallback に退行する(上位は decisions_with_zero_sample_actions で拒否できる)。
            chosen = sorted(actions, key=lambda action: action.sort_key())[0]
        else:
            values = dict((action, totals[action] / counts[action])
                          for action in sampled)
            chosen = sorted(sampled, key=lambda action: (-values[action],
                                                          action.sort_key()))[0]

        gap_counts = [counts[action] for action in actions]
        gap = max(gap_counts) - min(gap_counts)
        zeros = sum(1 for count in gap_counts if count == 0)
        self.stats.max_sample_gap.append(gap)
        self.stats.zero_sample_actions.append(zeros)
        if zeros:
            self.stats.decisions_with_zero_sample_actions += 1
        if self._budget_seconds is None and (zeros or gap > 1):
            raise AssertionError(
                "PIMC transition budget left unbalanced samples: "
                "max_sample_gap=%d zero_sample_actions=%d" % (gap, zeros))

        self.stats.decisions += 1
        self.stats.transitions += budget.spent
        self.stats.determinizations += passes
        decision_ms = budget.elapsed_ms
        self.stats.elapsed_ms.append(decision_ms)
        self.stats.search_elapsed_ms += decision_ms
        merge_deadline_overruns(budget, self.stats)
        self.stats.samples_per_action.append(
            dict((str(action.sort_key()), counts[action]) for action in actions))
        self.stats.transitions_per_action.append(
            dict((str(action.sort_key()), transitions[action]) for action in actions))
        return chosen
