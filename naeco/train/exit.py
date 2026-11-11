"""Expert Iteration の蒸留学習(Phase 7 段2/4)の公開 I/F。

**torch は import しない。** 実体は `naeco/train/ppo.py` の Expert Iteration 節にあり
(torch を import してよいのは `net.py` と `ppo.py` だけ、という Phase 3 設計書 §2 /
CLAUDE.md の規約を守るため)、本モジュールは名前を再輸出する facade である。
`scripts/train_exit.py` と `tests/test_exit.py` はこの facade を使う。

- 方策損失: 探索の改善方策 π'(または訪問分布)への交差エントロピー(合法手集合上)
- 価値損失: `(v_θ − z)^2`(z は記録席の終端効用 ±1。`value_mode="mix"` は根 value との平均)
- Policy surprise weighting(任意)
"""
from naeco.train.ppo import (   # noqa: F401
    TARGET_VERSION, ExitDataset, build_batch, evaluate, load_net, losses, train,
)

__all__ = ["TARGET_VERSION", "ExitDataset", "build_batch", "evaluate",
           "load_net", "losses", "train"]
