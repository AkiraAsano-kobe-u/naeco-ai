# naeco-ai

ボードゲーム『ナショナルエコノミー』(無印)の対戦AI。ルールエンジン、PPO 自己対戦の学習コード、ISMCTS 探索、学習済み重みを含みます。

> **非公式の実装です。** 原作は[スパ帝国](http://sparkigo.jp/)の『ナショナルエコノミー』で、本リポジトリは原作者・発売元と無関係です。カード画像・ルールブック本文は含みません。ルールの解釈に誤りがある可能性があります(採用した裁定は `RULES.md`)。

## 中身

| パス | 内容 |
|---|---|
| `naeco/engine/` | ルールエンジン(2〜4人、標準ライブラリのみ、seed 決定論・リプレイ対応) |
| `naeco/train/` | 観測・行動エンコーダ、方策価値ネット、PPO 自己対戦、チェックポイント |
| `naeco/ai/` | ヒューリスティック、SO-ISMCTS / PIMC 探索、評価ハーネス、価値の較正 |
| `data/cards.json` | カード定義(宣言的データ) |
| `docs/eval-runs/` | 学習済み重みと、その評価記録 |
| `scripts/` | 学習・評価・ラダーの CLI |

## 学習済み重み

| 人数 | ファイル | 記録されている成績 |
|---|---|---|
| 2人 | `docs/eval-runs/phase7-champion/gen1300.*` | 探索あり同士で旧版 gen532 に 1575勝25敗(800ペア、`docs/eval-runs/phase8/accept-0-799.json`) |
| 3人 | `docs/eval-runs/phase10/models/n3/gen199.*` | 探索ありでヒューリスティック2体に 2312/2400 勝(`docs/eval-runs/phase10/models/champion.json`) |
| 4人 | `docs/eval-runs/phase10/models/n4/gen199.*` | 探索ありでヒューリスティック3体に 3082/3200 勝(同上) |

`phase3-final-checkpoint/gen532` は 2人戦の旧版で、比較用に残しています。

## 使い方

CPython 3.9 で開発・検証しています。エンジンは標準ライブラリだけで動き、重みを使う場合は PyTorch (CPU) が必要です。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
.venv/bin/python -m pip install pytest

# 学習済みモデル・ヒューリスティック・ランダムの総当たり(2人戦、20ペア)
.venv/bin/python scripts/ladder.py \
  checkpoint:docs/eval-runs/phase7-champion/gen1300.meta.json heuristic random \
  --group train --start-index 0 --pairs 20 --workers 4 --out /tmp/ladder.json --execute

# 探索ありで対戦させる場合は search:<meta>@<予算>
#   search:docs/eval-runs/phase7-champion/gen1300.meta.json@4500

# 自己対戦学習
.venv/bin/python scripts/train_ppo.py --checkpoint-dir runs/demo --generations 2 --episodes 64 --workers 4

# テスト
.venv/bin/python -m pytest -q
```

## 注意

- 非公開の開発リポジトリから AI 部分を切り出したものです。Web UI と計算機ジョブ管理スクリプトは含みません。コメントや `docs/eval-seeds.md` が参照する `PLAN.md`・`docs/phase*-design.md` などの設計文書も含まれていません。
- `data/cards.json` の `source_refs`(SRC-xxx)と `RULES.md` の出典表記が指す調査ノート、および `champion.json` が参照する 3人・4人戦の合格判定の対局記録も含めていません。
- 評価記録中の実行パスは、ユーザー名部分を `/home/user` に置き換えています。それに合わせて、置き換えたファイルを指す SHA-256 の記載も更新しています。

## ライセンス

MIT(`LICENSE`)。『ナショナルエコノミー』の名称・ゲーム内容に関する権利は原作者に帰属します。
