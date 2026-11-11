# 評価 seed 台帳

この台帳は評価を開始する前に、判定条件と使用する index 範囲を固定する。
開発中の方策調整は `train` 群だけで行い、以下の `dev` 範囲での判定は一回だけ実行する。
追加判定が必要になった場合は未使用 index 範囲を新規に宣言する。

## ⚠ 2026-08-10 事後是正: 検出力設計は 2026-07-22 実行後に行った

Phase 2.5 の方式選定(2026-07-21 初回・2026-07-22 再判定)は、
**検出力設計を実行前に登録せずに実施された**。SOL-P25G-007 はこれを「観測後の
効果量を根拠に事前の検出力設計を省略できない」と指摘し、core-api §10.4 の必須手続き違反とした。

これを受けて **2026-08-10** に `scripts/power_search.py` を実装し、下記「再検証の事前登録」を
結果を見る前に追加した。これは**2026-08-10 時点の『事後是正』**であり、
**2026-07-22 の再判定結果を「事前登録済みの検出力設計で実行した」とは偽らない**。
2026-07-22 の採用(IS-MCTS)は T=4500 という限定条件での履歴判定として保持し、
再検証が完了して初めて正式採用の根拠を更新できる。

## 判定宣言: Phase 2 — vs ランダム

| 項目 | 固定値 |
|---|---|
| 対象 | `HeuristicAgent` vs `RandomAgent`、2人戦・席交換ペア |
| 閾値 | 90% |
| 対立仮説 `p_alt` / Δ | 97% / 7pt |
| 採用ペア数 | 500 (1000局) |
| seed group / index 範囲 | `dev` / 0–499 |
| bootstrap | ペア単位、B=10,000、seed=`2026072102` |
| 合格規則 | 95% CI 下限 **> 0.90**（厳密不等号） |

### 判定結果（2026-07-21 実行） — ⚠ **失効(SUPERSEDED)**

> **この判定は 2026-07-21 のエンジン修正により失効した。合否の根拠として使ってはならない。**
>
> 失効の理由: 開始プレイヤーが席0固定から seed 由来の抽選になった
> (RUL-SETUP-001 準拠。`naeco/engine/game.py` の `new_game`)。これにより
> 同じ `env_seed` でも **開始席・初期所持金・同点時のタイブレーク**が変わり、
> 下表の 1000 局は現行エンジンでは再現しない。`HeuristicAgent` の採石場評価も
> 変更しているため、方策自体も当時のものではない。
>
> 再判定は PLAN.md §5 に従い、**未使用の index 範囲を新規に宣言してから**
> 500ペア以上で実施すること。`dev` 0–499 は消費済みとして再利用しない。
> 新しい対応表には `start_player` / `initial_money` / `api_version` /
> `rules_version` を含める(`GameRecord` に追加済み)。

事前登録した設計（500ペア・閾値・bootstrap seed）は判定前に固定し、**結果に応じて
ペア数や範囲を変更していない**（逐次多重検定なし）。

| 項目 | 値 |
|---|---|
| 勝 / 分 / 敗 | 999 / 0 / 1（1000局） |
| 点推定 | 0.9990 |
| 95% CI | [0.9970, 1.0000] |
| 合否（CI下限 > 0.90） | **合格** |
| 検証環境 | CPython 3.14.4 と 3.9.25 で**同一数値**（環境非依存の決定論を確認） |

実行コマンド:
`python scripts/evaluate.py --pairs 500 --group dev --start-index 0 --records docs/eval-runs/phase2-dev-0-499.jsonl`

対応表(PLAN.md §5「seed・席・初期山札の対応表」)は
[`eval-runs/phase2-dev-0-499.jsonl`](eval-runs/phase2-dev-0-499.jsonl)(1000行)に永続化。
各行に以下を記録する:

- `group` / `index`、`env_seed` / `policy_seed` / `opponent_seed`
- `seat_variant`、`target_seat`(評価対象の席)、`opponent_seat`
- **`initial_deck`**(順序付き instance_id、58枚)、**`initial_hands`**(席別の初期手札)
  — 初期山札そのものを明示保存しており、カードデータや実装が将来変わっても失われない
- `initial_state_hash`(上記の整合性検証用の正準指紋)
- `cards_sha256` / `cards_data_version`(どのカードデータ版で生成した配札か)
- `result`(評価対象から見た 1.0 勝 / 0.5 分 / 0.0 敗)

唯一の敗北は dev index 379(seat_variant 0)。同ペアの seat_variant 1 は勝ちで、
両局の `initial_deck` / `initial_hands` は一致する(ペア=同一配札の席交換)。

## 判定宣言: Phase 2 再判定 — vs ランダム(2026-07-21 修正後)

**この宣言は実行前に固定した。以下を一切変更せずに1回だけ実行すること。**
結果に応じてペア数・index 範囲・閾値・bootstrap 設定を変更してはならない(逐次多重検定の禁止)。

| 項目 | 固定値 |
|---|---|
| 対象 | `HeuristicAgent` vs `RandomAgent`、2人戦・席交換ペア |
| 閾値 | 90% |
| 採用ペア数 | 500 (1000局) |
| seed group / index 範囲 | `dev` / **1000–1499**(未使用) |
| bootstrap | ペア単位、B=10,000、seed=`2026072102` |
| 合格規則 | 95% CI 下限 **> 0.90**（厳密不等号） |
| 前提エンジン | 開始プレイヤー抽選あり(RUL-SETUP-001)、`OBS_VERSION` 0.2.0 |

実行コマンド:

```
python scripts/evaluate.py --pairs 500 --group dev --start-index 1000 \
    --records docs/eval-runs/phase2-dev-1000-1499.jsonl
```

### 判定結果(2026-07-21 実行) — **合格**

宣言(ペア数・index範囲・閾値・bootstrap seed)は実行前に固定し、**結果を見てから
何も変更していない**。実行は1回のみ(逐次多重検定なし)。

| 項目 | 値 |
|---|---|
| 勝 / 分 / 敗 | **997 / 0 / 3**(1000局) |
| 点推定 | 0.9970 |
| 95% CI | **[0.9930, 1.0000]** |
| 合否(CI下限 > 0.90) | **合格** |
| プロセス終了コード | 0 |
| 検証環境 | CPython 3.14.4 と 3.9.25 で**同一数値**(環境非依存の決定論を確認) |

- 対応表: [`eval-runs/phase2-dev-1000-1499.jsonl`](eval-runs/phase2-dev-1000-1499.jsonl)(1000行)
- 敗北3局: index 1147(seat_variant 0)/ 1311(0)/ 1458(1)。いずれも同ペアのもう1局は勝ち
- 開始席の分布: `start_player` 0 が502局・1 が498局(RUL-SETUP-001 の抽選が効いている)
- 記録項目は旧対応表に加えて `setup_seed` / `start_player` / `initial_money` /
  `api_version`(0.1.0)/ `rules_version`(0.3.0)を含む
- 実行前に全テスト **231 passed + 30 subtests** を確認

## 版の注記: `rules_version` 0.3.0 → 0.3.1(2026-07-21、Phase 2.5)

Phase 2.5 の設計レビュー(sol P25-019)により、`OPEN-027`(他家の手札の消費財枚数を
**derived-public** に区分)は `RUL-INFO-001` が許す情報集合を変えるため、
`RULES_VERSION` を **0.3.1** へ改訂した。`rules_version` は `serialize` ヘッダに載るため
`state_hash` が変わる。

**上記2つの対応表(`phase2-dev-0-499.jsonl` / `phase2-dev-1000-1499.jsonl`)は
「rules 0.3.0 / OBS 0.2.0 / OPEN-027 以前」の歴史的成果物としてそのまま保持する。**
書き換えない。`initial_state_hash` は当時の版の指紋であり、現行エンジンでは再現しない。

**判定そのものは失効しない。** 0.3.1 で dev 1000–1499 を再実行して全1000行を突き合わせた結果:

| 項目 | 結果 |
|---|---|
| 勝 / 分 / 敗 | **997 / 0 / 3**(点推定 0.9970、95%CI [0.9930, 1.0000])— 完全一致 |
| `result` / `initial_deck` / `initial_hands` / `start_player` / `initial_money` | **全1000行一致** |
| 差分のあったフィールド | `rules_version` と `initial_state_hash` の**2つだけ**(各1000行、版変更による期待差分) |

したがって Phase 2 の合否根拠(配札・席・勝敗)は不変である。
今後の判定(Phase 2.5 以降)は 0.3.1 として**未使用 index 範囲**に新規保存する。

## 判定宣言: Phase 2.5 — 探索方式の選定(2026-07-21)

**この宣言は実行前に固定した。以下を一切変更せずに1回だけ実行すること。**
結果に応じてペア数・index 範囲・予算・パラメータ・採用基準を変更してはならない。

### 固定するパラメータ(段階1の結果から確定)

| 項目 | 固定値 | 根拠 |
|---|---|---|
| 計算予算 `T` | **4500**(1決定点あたりの状態遷移数) | 段階1。PIMC は T=1500 で飽和するが IS-MCTS は伸び続けるため、低い予算を選ぶと系統的に PIMC 有利になる。T=4500 は PLAN Phase 5 の思考時間目標(平均2秒 / p95 5秒)にも収まる(実測 平均1.1秒 / p95 2.0秒) |
| `H`(切詰め地平) | **2** | 下記「座標降下を採らなかった理由」 |
| `SCALE` | **30** | 同上 |
| `c`(UCB 係数) | **1.41** | 段階1で既定が最良(0.7→0.900、2.0→0.920、1.41→0.940) |

**⚠ 座標降下の結果を採らなかったのは「事前選択規則からの事後的逸脱」である**
(sol SOL-P25G-006 による訂正)

事前登録した規則は「点推定が最大の組を選ぶ」であり、その手続きどおりに走査した結果は
`H=1` / `SCALE=50` だった。**結果を見たあとでこの規則を撤回し、既定値へ戻している。**
初版はこれを「規則外の判断を持ち込まずに済んだ」と記述したが、**その記述は誤りである**。

- **逸脱の判断時点**: 段階1(`train` 群)の結果を見た後、**段階2(`dev`)を実行する前**。
  dev のデータは一切見ていないので、判定群への漏洩は生じていない。
- **逸脱の理由**: (a) 全候補の 95%CI が大きく重なり**統計的に支持された改善が一つもない**、
  (b) 組合せ `H=1 × SCALE=50` を train で確認したところ **PIMC が 0.950 → 0.920 と悪化**し
  (IS-MCTS は 0.970 → 0.980)、座標降下が仮定した**分離可能性が成り立たなかった**、
  (c) `SCALE=50` は片方の方式にだけ有利に働くため**方式比較が偏る**。
- **事前規則の欠陥**: `SCALE` は 20 と 50 が同点(両方式平均 0.965)だったが、
  事前規則はこの同点を想定していなかった。また「組合せ確認」「不確実なら既定維持」も
  規則に含めていなかった。**今後の事前登録にはこの3点を含める。**

**train 群での測定コマンド**(初版の台帳は index を誤記していた。実データは
`start_index=0` の 0–49。RSS 計測のみ 900–902 を使い、これは JSONL へ記録していない):

```
# 予算感度と座標降下(16構成)
python scripts/compare_search.py --stage 1 --pairs 50 --budgets 500 1500 4500     --workers 12 --records docs/eval-runs/phase2.5-stage1-train.jsonl --execute
# 組合せ確認(H=1, SCALE=50)は run_comparison を直接呼んで実施し、
# 同ファイルへ label="combined-H1-S50" として追記した。現行 CLI 経路では再現しない。
```

### 判定条件

| 項目 | 固定値 |
|---|---|
| 対戦 | ① `PIMC vs Heuristic` ② `IS-MCTS vs Heuristic` ③ `PIMC vs IS-MCTS` |
| 採用ペア数 | **各対戦 500ペア(1000局)** |
| seed group / index 範囲 | `dev` / **2000–2499**(未使用) |
| bootstrap | ペア単位、B=10,000、seed=`2026072102`(Phase 2 と同一) |
| 前提バージョン | `rules_version` 0.3.1 / `OBS_VERSION` 0.3.0 / `API_VERSION` 0.1.0 |

**p 値**: 席交換ペアの設計により、探索なし(Heuristic 同士)の期待勝率は厳密に 0.5 である。
よって帰無仮説は「ペア勝率 = 0.5」、対立仮説は片側「> 0.5」。

```
p = (1 + #{ b ∈ 1..B : mean(bootstrap標本 b) − 0.5 <= 0 }) / (B + 1)
```

**採用基準(Holm 補正つき状態機械。片側 α = 0.05)**

```
primary family = {①, ②}
  p を昇順に並べ p_(1) <= p_(2) として
    p_(1) <= α/2  なら棄却し、続けて p_(2) <= α なら棄却する
    (p_(1) > α/2 の時点で両方とも棄却しない)

棄却0件 → 「探索なし」を採用(この計算予算では探索の優位を示せなかったと結論)
棄却1件 → その方式を採用
棄却2件 → ③ を独立した片側 α=0.05 の secondary test として判定
            p3 <= 0.05 → 勝った方式
            p3 >  0.05 → タイブレーク: p95 思考時間が小さい方
                          → 同値ならピーク RSS が小さい方
                          → それも同値または未計測なら **PIMC**
```

### 実行コマンド

```
python scripts/compare_search.py --stage 2 --pairs 500 --budget 4500 \
    --horizon 2 --scale 30 --c 1.41 --workers 12 \
    --records docs/eval-runs/phase2.5-stage2-dev-2000-2499.jsonl --execute
```

判定は記録された `pair_scores` から `scripts/decide_search.py` で決定論的に算出する
(実行と判定を分け、結果を見てから基準を動かせないようにする)。

所要見込み: ①約26分 / ②約37分 / ③約63分(12並列)= **合計 約2.1時間**。

### 判定結果(2026-07-21 実行) — **IS-MCTS を採用**

宣言(ペア数・index範囲・予算・パラメータ・採用基準)は実行前に固定し、
**結果を見てから何も変更していない**。実行は1回のみ(逐次多重検定なし)。
実所要 1時間54分(12並列)。

| 対戦 | 勝率 | 95%CI | 勝/分/敗 |
|---|---:|---|---:|
| ① PIMC vs Heuristic | 0.9760 | [0.9660, 0.9850] | 976/0/24 |
| ② IS-MCTS vs Heuristic | 0.9600 | [0.9480, 0.9720] | 960/0/40 |
| ③ PIMC vs IS-MCTS | **0.3380** | [0.3090, 0.3680] | 338/0/662 |

事前登録した状態機械の出力:

```
primary p値: pimc=0.00010 ismcts=0.00010 (片側 α=0.05、Holm) → 2件とも棄却
secondary(pimc vs ismcts) p=1.00000 / 逆向き p=0.00010 → IS-MCTS が有意
採用方式: ismcts
```

**★ 非推移性**: 共通相手(Heuristic)には PIMC が上(0.976 > 0.960)なのに、
直接対決では IS-MCTS が 66.2% で勝つ。共通相手だけで選定していれば誤っていた。
詳細と全計測値は [`phase2.5-search-comparison.md`](phase2.5-search-comparison.md)。

## 判定宣言: Phase 2.5 **再判定** — 探索方式の選定(2026-07-22)

**この宣言は実行前に固定した。以下を一切変更せずに1回だけ実行すること。**

### なぜ再判定するか

solゲート(`docs/reviews/2026-07-21-sol-phase2.5-gate.md`)で **SOL-P25G-002** を受けた。
初回(dev 2000–2499)は**対戦ごとの集計3行しか保存しておらず**、PLAN.md §5 が要求する
「seed・席・初期山札の対応表」が無かった。保存物から席の均衡・同一配札・seed 対応・
版固定を独立に監査できないため、**判定としては採用しない**。
初回の記録は参考値として残すが、合否の根拠には使わない。

あわせて **SOL-P25G-004** を修正した。初回の宣言は ③ の直接対決を
「順方向と逆方向をそれぞれ片側 α=0.05」で判定しており、方向を事前固定していない以上
**第一種過誤が最大 0.10 まで膨らむ**。本宣言では**両側 α=0.05** とする。

### 固定するパラメータ(初回から変更しない)

| 項目 | 固定値 |
|---|---|
| 計算予算 `T` | **4500**(1決定点あたりの状態遷移数) |
| `H` / `SCALE` / `c` | **2 / 30 / 1.41** |

段階1(train)の結果と選定根拠は初回宣言のとおりで、**再取得しない**
(train 群での探索は既に済んでおり、dev を見てから変えていない)。

### 判定条件

| 項目 | 固定値 |
|---|---|
| 対戦 | ① `PIMC vs Heuristic` ② `IS-MCTS vs Heuristic` ③ `PIMC vs IS-MCTS` |
| 採用ペア数 | **各対戦 500ペア(1000局)** |
| seed group / index 範囲 | `dev` / **2500–2999**(未使用) |
| bootstrap | ペア単位、B=10,000、seed=`2026072102` |
| 前提バージョン | `rules_version` 0.3.1 / `OBS_VERSION` 0.3.0 / `API_VERSION` 0.1.0 |
| **対応表** | **1局1行**(index / seat_variant / 各方式の席 / 結果 / 最終得点 / 全seed / `initial_deck` / `initial_hands` / `initial_money` / `start_player` / `initial_state_hash` / `cards_sha256` / 各版 / 予算とパラメータ) |

**p 値**(初回と同一): 席交換ペアなので探索なし同士の期待勝率は厳密に 0.5。

```
p = (1 + #{ b : mean(bootstrap標本 b) - 0.5 <= 0 }) / (B + 1)
```

**採用基準(Holm 補正つき状態機械。片側 α=0.05、③のみ両側)**

```
primary family = {①, ②}
  p を昇順に並べ p_(1) <= p_(2) として
    p_(1) <= α/2 なら棄却し、続けて p_(2) <= α なら棄却する

棄却0件 → 「探索なし」を採用
棄却1件 → その方式を採用
棄却2件 → ③ を **両側 α=0.05** で判定(SOL-P25G-004 の修正)
            p_two_sided = min(1, 2 * min(p_順方向, p_逆方向))
            p_two_sided <= 0.05 → 点推定の符号で勝った方式
            p_two_sided >  0.05 → タイブレーク: p95思考時間 → ピークRSS → PIMC
```

### 実行コマンド

```
python scripts/compare_search.py --stage 2 --pairs 500 --budget 4500     --horizon 2 --scale 30 --c 1.41 --workers 12 --start-index 2500     --records docs/eval-runs/phase2.5-stage2b-dev-2500-2999.jsonl --execute
```

対応表は `...-games.jsonl` に自動で併記される。記録先が既に存在する場合、
スクリプトは**実行を拒否する**(追記による証跡の汚染を防ぐ)。
判定は `scripts/decide_search.py` で決定論的に算出する。

#### 実行の中断と継続(2026-07-22、記録)

上記コマンドは **①② を完走した直後(開始から61分)に外部から強制終了された**。
Python の例外ではない(`2>&1` を付けていたがトレースバックが皆無で、終了コードは 4)。
原因は**実行時間の上限**と判断した: ①1582秒 + ②2102秒 に ③(両側が探索なので
①+② 相当の約61分)を足すと**総計 約123分**となり、2時間を超える。
直前に成功した初回実行は 114分で、上限に触れていなかった。
③ を単体で計測すると1ペア 79〜95秒・ピーク RSS 27MB で正常に完走するため、
③ のロジックには問題がない。

**継続の扱い**: ①② の記録(各1000行)は完全であることを検査した
(index 2500–2999 の500種、席分布 500/500、ペア内で `initial_deck` と
`initial_state_hash` が一致、版が単一)。**①② は再実行せず**、③ のみを
`--matchups pimc-vs-ismcts --resume` で追加実行した。同じ対戦を複数回実行して
良い結果を選ぶことはしていない(`--resume` は記録済みラベルを自動でスキップする)。
seed・予算・パラメータは宣言から一切変更していない。

③ の追加実行は 2026-07-22 13:14〜14:30(4549秒)。バッテリー稼働・12並列で完走した。

### 判定結果(2026-07-22 実行) — **IS-MCTS を採用**

| 対戦 | 勝率 | 95%CI | 勝/分/敗 |
|---|---:|---|---:|
| ① PIMC vs Heuristic | 0.9750 | [0.9650, 0.9840] | 975/0/25 |
| ② IS-MCTS vs Heuristic | 0.9640 | [0.9520, 0.9750] | 964/0/36 |
| ③ **PIMC vs IS-MCTS** | **0.3570** | [0.3280, 0.3860] | 357/0/643 |

`scripts/decide_search.py` の出力(事前登録した状態機械):

```
primary p値: pimc=0.00010 ismcts=0.00010 (片側 α=0.05、Holm) → 2件とも棄却
secondary(pimc vs ismcts) 片側 p: 順方向=1.00000 逆方向=0.00010
                          → 両側 p=0.00020 (α=0.05) 点推定=0.3570
採用方式: ismcts
```

**対応表(3000行)を保存物だけから検査**: 各対戦1000行、index 2500–2999 の500種、
席分布 500/500、ペア内で `initial_deck` と `initial_state_hash` が一致、
版(rules 0.3.1 / obs 0.3.0 / api 0.1.0)と予算 `(4500, H=2, SCALE=30, c=1.41)` が単一。

**★ 共通相手ランキングの反転**: 共通相手(Heuristic)には PIMC が上(0.975 > 0.964)
なのに直接対決では IS-MCTS が 64.3% で勝つ。共通相手だけで選定していれば誤っていた。
※ 勝敗関係は `IS-MCTS > PIMC > Heuristic` の全順序であり**非推移性ではない**
(sol SOL-P25G-010)。

初回(dev 2000–2499、対応表なし)とは3対戦とも CI がほぼ完全に重なる。
詳細は [`phase2.5-search-comparison.md`](phase2.5-search-comparison.md)。

## 使用済み index 範囲

| 日付 | 用途 | group | index 範囲 | 状態 |
|---|---|---|---|---|
| 2026-07-21 | Phase 2 最終判定（vs ランダム） | dev | 0–499 | **消費済み・失効(SUPERSEDED)** |
| 2026-07-21 | Phase 2 **再判定**(バグ修正後) | dev | **1000–1499** | **消費済み・有効(合格)** |
| 2026-07-21 | Phase 2.5 探索方式の選定(初回) | dev | **2000–2499** | **消費済み・参考値**(solゲート SOL-P25G-002 により対応表不足。合否根拠に使わない) |
| 2026-07-22 | Phase 2.5 探索方式の選定(**再判定**) | dev | **2500–2999** | **消費済み・有効(IS-MCTS 採用)** |
| 2026-07-21 | Phase 2.5 予算感度・座標降下・組合せ確認 | train | **0–49**(全走査。RSS 計測のみ 900–902) | 開発用(判定ではない) |
| 2026-08-10 | Phase 2.5 N01修正後 公式再実行(a41 run) | dev | **4000–4784** | **消費済み・最終ゲート証拠にしない**(完走・IS-MCTS 採用。DeepSeek 監査で source digest 変更が必要) |
| 2026-08-10 | 同上 wall-clock 交差確認 | train | **400–599** | **消費済み・最終ゲート証拠にしない** |
| 2026-08-10 | 同上 parallel 決定論 | train | **600–607** | **消費済み・最終ゲート証拠にしない**(一致実測はあるが修正後の再実測が必要) |
| 2026-08-10 | Phase 2.5 最終 corrective v4 run(開始→TERM) | dev | **5000–5784** | **消費済み・最終ゲート証拠にしない**(cross audit により全 job TERM) |
| 2026-08-10 | 同上 wall-clock 交差確認 | train | **700–899** | **消費済み・最終ゲート証拠にしない**(全 job TERM) |
| 2026-08-10 | 同上 parallel 決定論 | train | **900–907** | **消費済み・最終ゲート証拠にしない**(900–902 は既存 RSS 計測(train 900–902)と衝突。全 job TERM) |

`dev` 0–499 は消費済みとして扱い、**再利用しない**。2026-07-21 のエンジン修正
(開始プレイヤーの抽選化・採石場評価の変更)により当時の 1000 局は現行エンジンで
再現しないため、この範囲の合否は失効している(上記の警告ブロックを参照)。
再判定は未使用範囲 dev 1000–1499 を事前宣言して実施済み(合格)。dev 1000–1499 も
消費済みであり、今後さらに判定が必要になった場合は**さらに未使用の範囲**を新規宣言する。

## 判定宣言: Phase 2.5 **再検証の事前登録**(2026-08-10)

> **この宣言は結果を見る前に固定した(2026-08-10)。** SOL-P25G-007 への事後是正として、
> 未実行の再検証を「事前登録済み」の手続きで宣言した。**dev 3×785・parallel 決定論・
> wall-clock 交差確認は 2026-08-10 にすべて正式実行済み**であり、下記に実数値を記入する。

### 固定する条件(2026-07-22 の履歴判定と同一)

| 項目 | 固定値 |
|---|---|
| 計算予算 `T` | **4500**(1決定点あたりの状態遷移数) |
| `H` / `SCALE` / `c` | **2 / 30 / 1.41** |
| 対戦 | ① PIMC vs Heuristic ② IS-MCTS vs Heuristic ③ PIMC vs IS-MCTS |
| bootstrap | ペア単位、B=10,000、seed=`2026072102` |
| 前提バージョン | `rules_version` 0.3.1 / `OBS_VERSION` 0.3.0 / `API_VERSION` 0.1.0 |
| 対応表 | 1局1行(index / seat_variant / 各方式の席 / 結果 / 最終得点 / 全seed / `initial_deck` / `initial_hands` / `initial_money` / `start_player` / `initial_state_hash` / `cards_sha256` / 各版 / 予算とパラメータ) |

### 検出力設計(SOL-P25G-007)

- **実用最小差 δ = 0.05**、**secondary は両側 α=0.05**、**検出力 ≥ 0.80** を保守基準にする。
- σ は既存 train JSONL `docs/eval-runs/phase2.5-stage1-train.jsonl` の
  `pair_scores` 標本標準偏差(ddof=1)から取得する。**JSON 破損・値不正
  ([0,1] 外・非数・`pair_scores` 欠落・空)はエラー拒否**とし、fallback しない。
  **該当行なし・標本不足(1個以下)の場合のみ** [0,1] score の保守上限 σ=0.5 へ
  fallback する(コードの契約に合わせた統一)。
- 必要ペア数は `scripts/power_search.py` が ceil で決定論的に算出する。
  各 family の算出 N は 73 / 46 / 785。
- **3対戦を同一 run/config に統一するため、全 family で最大値 785ペアを採用**
  (PLAN.md §5 の 500 floor も満たす)。③ は既定構成 T=4500 の train 直接対決記録が
  存在しないため σ=0.5 へ fallback したことが最大値の根拠になる。

**2026-08-10 に CLI で得た実数(根拠コマンドとともに固定):**

```
python scripts/power_search.py --delta 0.05 --power 0.80 \
    --records docs/eval-runs/phase2.5-stage1-train.jsonl \
    --label default --budget 4500 --matchup pimc-vs-heuristic
# → σ=0.1515, required_pairs=73

python scripts/power_search.py --delta 0.05 --power 0.80 \
    --records docs/eval-runs/phase2.5-stage1-train.jsonl \
    --label default --budget 4500 --matchup ismcts-vs-heuristic
# → σ=0.1199, required_pairs=46

python scripts/power_search.py --delta 0.05 --power 0.80 \
    --records docs/eval-runs/phase2.5-stage1-train.jsonl \
    --label default --budget 4500 --matchup pimc-vs-ismcts
# → 該当なし → σ=0.5 fallback, required_pairs=785
```

| 対戦 | σ の根拠 | 算出 N | **採用 N** |
|---|---:|---:|---:|
| ① PIMC vs Heuristic | train default T=4500 標本 σ=0.1515 | 73 | **785** |
| ② IS-MCTS vs Heuristic | train default T=4500 標本 σ=0.1199 | 46 | **785** |
| ③ PIMC vs IS-MCTS | 該当なし → fallback σ=0.5 | 785 | **785** |

> 全 family の採用 N は算出 N の最大値 785 へ統一する。①・② は算出値 73 / 46 を
> 下回らないよう、785 をそのまま採用する。

### 使用する seed 範囲(事前予約)

`dev` 3000 以降を未使用範囲として予約する。既存使用済みは dev 0–499 / 1000–1499 /
2000–2499 / 2500–2999 であり、**3000 以降はどの既存使用範囲とも衝突しない**。

| 対戦 | 採用 N | dev range | 状態 |
|---|---:|---|---|
| ① PIMC vs Heuristic | 785 | **3000–3784** | **消費済み・N01修正前(再ゲートの採用証拠にしない)** |
| ② IS-MCTS vs Heuristic | 785 | **3000–3784** | **消費済み・N01修正前(再ゲートの採用証拠にしない)** |
| ③ PIMC vs IS-MCTS | 785 | **3000–3784** | **消費済み・N01修正前(再ゲートの採用証拠にしない)** |

### 壁時計交差確認に使う train 範囲(事前予約)

段階1は train 0–49(および RSS 計測のみ 900–902)を使用済み。**train 100–299 を
未使用として予約**し、同一 wall-clock 予算での交差確認(SOL-P25G-003)に使う。
これは**主判定ではなく交差確認**であり、統計的採用基準の変更を意味しない。
wall-clock 交差確認は **2026-08-10 に正式実行済み**。結果は下記の検査表へ記入し、
`train` 100–299 は **消費済み**とする。**ただしこれは N01 修正前 digest の参考であり、
再ゲートの採用証拠にはしない。**

**wall-clock 交差確認の固定条件(結果を見る前の宣言):**

| 項目 | 固定値 |
|---|---|
| group / pairs / range | `train` / **200ペア** / **100–299**(未使用) |
| 予算 | **budget_seconds=2.0/決定点**、transition safety cap=**1,000,000** |
| パラメータ | `H=2` / `SCALE=30` / `c=1.41` |
| 対戦 | ① PIMC vs Heuristic ② IS-MCTS vs Heuristic ③ PIMC vs IS-MCTS |
| workers | **96**(csi-enshu 向けの運用値) |
| cap の意味 | `budget=1,000,000` は独立な別機構ではなく、**wall-clock 予算時の transition
  budget 上限値**。`--budget-seconds` 使用時は CLI が `--budget` の明示と
  `>= 1,000,000` を要求する |
| 判定 | 主判定と同じ **Holm primary(①②、片側 α=0.05)+ secondary 両側 α=0.05**
  の state machine を使い、**IS-MCTS 選択が維持されること** |
| 追加成功条件 | 全探索 method で `zero_sample_actions=0`、**期限超過量を保存できる**こと |

### parallel 決定論確認に使う train 範囲(事前予約)

**train 300–307(8ペア)を未使用として予約**し、SOL-P25G-005 の並列一致検査に使う。

| 項目 | 固定値 |
|---|---|
| group / pairs / range | `train` / **8ペア** / **300–307**(未使用) |
| 予算 | `T=4500`(1決定点あたりの状態遷移数) |
| パラメータ | `H=2` / `SCALE=30` / `c=1.41` |
| 対戦 | ① PIMC vs Heuristic ② IS-MCTS vs Heuristic ③ PIMC vs IS-MCTS |
| 比較 | **worker=1 vs worker=8** |
| 成功条件 | 全決定論的フィールド一致(局結果・`pair_scores`・全探索統計・対応表)、
  4 thread env の実効値がすべて `1`、JSON report を保存 |

この parallel 決定論は **2026-08-10 に正式実行済み**(結果は下記の検査表)。
**ただしこれは N01 修正前 digest の参考であり、再ゲートの決定論証拠にはしない。**

### 成功判定(結果を見る前の宣言)

再検証は次の**すべて**を満たしたときに成功とする:

1. ①② が primary family(Holm、片側 α=0.05)で棄却され、探索なしより優位を示す
2. ③ が **両側 α=0.05** で有意(p_two_sided ≤ 0.05)かつ点推定の符号が **IS-MCTS 有利**
   (`pimc-vs-ismcts` の点推定 < 0.5)。**点推定 == 0.5 は方式勝利とせずタイブレークへ**
3. 3対戦とも採用 N=785 を欠損・重複なく完走し、対応表の席分布 785/785 ・
   ペア内同一配札・版の単一性を保存物だけから検査できる
4. 壁時計交差確認(train 100–299、200ペア、budget_seconds=2.0/決定点、
   safety cap=1,000,000)で主判定と同じ state machine の **IS-MCTS 選択が維持**され、
   全探索 method で `zero_sample_actions=0`・期限超過量の保存ができる
5. parallel 決定論(train 300–307、8ペア、worker=1 vs 8)で全決定論的フィールド一致、
   4 thread env=1、JSON report 保存ができる

**2026-08-10 実績**: **N01 修正前の実行では成功条件 1〜5 をすべて充足した。**
dev 3×785・parallel 決定論・wall-clock 交差確認をすべて正式実行済みとし、
下記の実数値で確定した。**N01 修正後の公式再ゲートは新範囲(dev 4000–4784 /
wallclock train 400–599 / determinism train 600–607)で実行済み(a41 run)だが、
完走後の DeepSeek 監査で source digest を変える必須修正が判明し、最終ゲート証拠に
しない。最終ゲート証拠は下記「Phase 2.5 最終 corrective 事前登録」の新範囲
(dev 5000–5784 / train 700–899 / 900–907)のみとする。**

> **2026-08-10 追記(v4 run の失効)**: 上記の最終 corrective(v4)自体も、開始後の
> cross audit で det 範囲 900–907 のうち 900–902 が既存 RSS 計測(train 900–902)と
> 衝突することが判明し、全 job を TERM した。3範囲(dev 5000–5784 / train 700–899 /
> 900–907)は消費済み・最終ゲート証拠外。真の最終事前登録は下記
> 「Phase 2.5 真の最終 corrective 事前登録」の dev 6000–6784 / train 1100–1299 /
> 1400–1407 のみ。

### 実行コマンド(固定)

**full dev compare(主判定):**

```
python scripts/compare_search.py --stage 2 --pairs 785 --budget 4500 \
    --horizon 2 --scale 30 --c 1.41 --workers 96 --start-index 3000 \
    --records docs/eval-runs/phase2.5-stage2c-dev-3000.jsonl --execute

python scripts/decide_search.py docs/eval-runs/phase2.5-stage2c-dev-3000.jsonl \
    --protocol phase2.5-dev
```

①②③ をこの同一コマンド・同一 run/config で実行する(記録先が存在する場合は
スクリプトが拒否する。追記による証跡の汚染を防ぐ)。`--workers 96` は
csi-enshu 向けの**運用条件**であり、統計条件(ペア数・α・power・seed 範囲)ではない。
判定は**新形式 protocol 必須**。公式の新形式は `phase2.5-dev` /
`phase2.5-wallclock` のみで、`custom` は存在しない。履歴成果物を読む場合のみ
従来どおり `--legacy` を明示する。

`stage2` は **`--start-index` 明示必須**。上記固定3コマンドはすべて明示済みである。

**wall-clock 交差確認(主判定ではない):**

```
python scripts/compare_search.py --stage 2 --group train --pairs 200 \
    --budget 1000000 --budget-seconds 2.0 --horizon 2 --scale 30 --c 1.41 \
    --workers 96 --start-index 100 \
    --records docs/eval-runs/phase2.5-wallclock-train-100-299.jsonl --execute

python scripts/decide_search.py docs/eval-runs/phase2.5-wallclock-train-100-299.jsonl \
    --protocol phase2.5-wallclock
```

CLI 事故防止(監査後の契約): `--budget-seconds` 使用時は `--budget` を明示し、
transition safety cap `1,000,000` 以上を要求する。`--check-determinism` 使用時は
`--pairs` の明示と既知 matchup のみを受け付ける(既定の全3 matchups も明示可能)。

**parallel 決定論(worker=1 vs 8、train 300–307):**

```
python scripts/compare_search.py --stage 2 --group train --pairs 8 \
    --budget 4500 --horizon 2 --scale 30 --c 1.41 --workers 8 \
    --start-index 300 --check-determinism \
    --records docs/eval-runs/phase2.5-determinism-train-300-307.json --execute
```

この実行は worker=1 と worker=8 を同一設定で比較し、成功時のみ上記 JSON report を
原子保存する。全3 matchups を既定のまま実行する。

### 判定結果(2026-08-10 正式実行済み)

> **この判定結果は N01 修正前 digest の参考**であり、N01 修正後の正式再ゲート
> (下記「Phase 2.5 N01 修正後の正式再実行の事前登録」)の採用証拠にはしない。

| 対戦 | pairs | dev range | 勝率 | 95%CI | 勝/分/敗 | 本 run 壁時計 |
|---|---:|---|---:|---:|---|---:|
| ① PIMC vs Heuristic | 785 | 3000–3784 | 0.973885 | [0.965605, 0.981529] | 1529/0/41 | 120.9147 s |
| ② IS-MCTS vs Heuristic | 785 | 3000–3784 | 0.959236 | [0.948408, 0.968790] | 1506/0/64 | 154.4476 s |
| ③ PIMC vs IS-MCTS | 785 | 3000–3784 | 0.337580 | [0.315287, 0.361146] | 530/0/1040 | 266.2880 s |

| 検査項目 | 結果 |
|---|---|
| primary 棄却(①②) | **p=0.00010 / 0.00010(Holm)→2件棄却** |
| secondary 両側 p(③) | **0.00020** |
| 採用方式 | **ismcts** |
| 対応表の完全性・決定論的フィールド | 各785ペア=1570行・席 785/785・index 3000–3784・ペア内 `initial_state_hash` 一致・bundle manifest / run_id / source_digest 照合一致 |
| `zero_sample_actions` / deadline | 全3 label で `decisions_with_zero_sample_actions=0`・`deadline_overrun_count=0` |
| nodes/s(artifact 転記) | ① PIMC **0.0**(非木方式)/ ② IS-MCTS **420.14682423199275** / ③ PIMC **0.0**・IS-MCTS **413.63140423175804** |
| worker=1 vs 8 並列一致(train 300–307, pairs=8, T=4500) | **3 labels 全決定論的 field 一致**・4 thread env 実効値すべて 1・JSON report 保存 |
| 壁時計交差確認(train 100–299, pairs=200, budget_seconds=2.0) | **正式実行済み**。下記「wall-clock 交差確認の判定結果」へ実数値を記入 |

**正式実行の環境と証跡**:

- 公式実行は **`/usr/bin/python3` = CPython 3.9.21**(csi-enshu、host `csi-hpc-01`)、
  source_digest **`b7fcc48b1bd639388f518524c9138d806822797db636e45b85683e0738577092`**、
  workers=96、thread env 4変数すべて `1`。
- `full-dev-py39` / `determinism-py39` / `tests-py39-v2` はいずれも **exit_code=0**。
  実行コマンドは `docs/eval-runs/phase2.5-csi-jobs-20260810/*/command` に保存。
- run_id: ① `94d5a6c68a6f6f2645c8266df51e1aa88b6a5557623b417f0e961e7020d5d56a` /
  ② `a916a5b42092ba0e4eca33c1fd74ae2106c124562c86078e1ce8709ce2fe5cb6` /
  ③ `990d0575689f144700a5cf3aacdeabfbe9589248b0df8d7882273618f7209d3f`。
- 記録: `docs/eval-runs/phase2.5-stage2c-dev-3000.jsonl`(bundle 3件・各 summary 1行・
  games 各1570行)、`docs/eval-runs/phase2.5-stage2c-dev-3000-decision.txt`、
  `docs/eval-runs/phase2.5-determinism-train-300-307.json`。
- テスト: **ローカル(`/usr/bin/python3` 3.9.6)と remote(csi-enshu 3.9.21)で 419 tests OK**。
- **無効実行**: `/ap/anaconda3` の **CPython 3.12.7** による初回実行は
  `docs/eval-runs/phase2.5-csi-jobs-20260810/invalid-py3127/` に隔離し、
  **公式結果から除外**。無効実行の数値は採用根拠に引用しない。

### wall-clock 交差確認の判定結果(2026-08-10 正式実行済み)

> **この判定結果は N01 修正前 digest の参考**であり、N01 修正後の正式再ゲートの
> 採用証拠にはしない。

固定条件どおり `train` 100–299・各 **200ペア(400局)**・`budget=1,000,000`
(transition budget 上限値)・`budget_seconds=2.0/決定点`・`H=2` / `SCALE=30` /
`c=1.41`・3対戦・**workers=96** で実行した。

| 対戦 | 勝率 | 95%CI | 勝/分/敗 | 本 run 壁時計 |
|---|---:|---:|---:|---:|
| ① PIMC vs Heuristic | **0.9925** | [0.9825, 1.0000] | 397/0/3 | 260.1146 s |
| ② IS-MCTS vs Heuristic | **0.9500** | [0.9300, 0.9700] | 380/0/20 | 238.1921 s |
| ③ PIMC vs IS-MCTS | **0.2700** | [0.2300, 0.3100] | 108/0/292 | 498.2266 s |

```
primary p値: pimc=0.00010 ismcts=0.00010 (片側 α=0.05, Holm) → 2件棄却
secondary(pimc vs ismcts) → 両側 p=0.00020 (α=0.05) 点推定=0.2700
→ 直接対決で有意差あり → 採用: ismcts
```

| 検査項目 | 結果 |
|---|---|
| 主判定と同じ state machine で IS-MCTS 選択が維持される | **維持(choice=ismcts)** |
| 全探索 method の `zero_sample_actions` | **3対戦全 method で 0** |
| deadline fields 保存 | **保存済み**。`deadline_overrun_count / total_ms / max_ms` を方式別に記録 |
| 対応表の完全性 | 各200ペア=400行・席 200/200・index 100–299・manifest 照合一致 |
| 環境 | `/usr/bin/python3` 3.9.21・source_digest `b7fcc48b...`・workers=96・thread env 4変数=1 |
| post job | `post-and-wallclock-py39` の **exit_code=0** |
| train 範囲 | **100–299 消費済み** |

**方式別 `search_metrics`(artifact 転記、③ は PIMC/ISMCTS 両方):**

| label / method | transitions/decision | truncated/evaluations 率 | mean think | p95 think | deadline count / total / max | nodes/s |
|---|---:|---:|---:|---:|---:|---:|
| ① PIMC | 31,328.06 | 0.0005516(7,872/14,270,347) | 2000.096 ms | 2000.217 ms | 8,981 / 463.922 ms / 0.652 ms | 0.0(非木方式) |
| ② IS-MCTS | 24,117.10 | 0.0008996(7,531/8,371,125) | 2000.123 ms | 2000.339 ms | 7,860 / 613.312 ms / 0.820 ms | 441.4799 |
| ③ PIMC | 31,243.57 | 0.0006257(8,429/13,471,260) | 2000.099 ms | 2000.215 ms | 9,455 / 494.721 ms / 0.614 ms | 0.0(非木方式) |
| ③ IS-MCTS | 25,287.15 | 0.0010396(7,396/7,114,263) | 2000.125 ms | 2000.343 ms | 7,691 / 598.218 ms / 3.920 ms | 414.8713 |

**cap 1,000,000 未到達の実測**: 各方式の transitions/decision は上表のとおりで、
いずれも **1,000,000 より大幅に小さく、transition budget 上限には達していない**。
wall-clock(2.0 s/決定点)が実効予算として支配的だったことを示す。

記録: `docs/eval-runs/phase2.5-wallclock-train-100-299.jsonl`(bundle 3件・各 summary 1行・
games 各400行)、`docs/eval-runs/phase2.5-wallclock-train-100-299-decision.txt`。
これは pre-N01 digest の歴史成果物であり、現行 strict protocol での再検証済みを意味しない。
再現は `decide_search.py --legacy` で行う。strict protocol の適用は下記
「Phase 2.5 真の最終 corrective 事前登録」の新rangeのみである。

## 判定宣言: Phase 2.5 N01 修正後の正式再実行の事前登録(2026-08-10)

> **⚠ 2026-08-10 完了後追記(a41 run の失効)**: 本節の3範囲(dev 4000–4784 /
> wallclock train 400–599 / determinism train 600–607)は事前登録どおり完走し、
> IS-MCTS 採用・並列決定論一致を確認した。ただし完走後の DeepSeek 監査で
> CI 再計算・label 左右整合・det report の runtime 保存・sampler 入力/csi 修正が
> 必須と判明し、source digest が変わるため、本 run(a41)は**最終ゲート証拠にしない**。
> seed 範囲はすべて**消費済み**。最終ゲート証拠は下記「Phase 2.5 最終 corrective
> 事前登録」の新範囲のみとする。

> **これは結果を見る前の宣言である。** N01(hash 非決定性)修正後の公式再実行を
> 新規 seed 範囲で行う。**結果欄はすべて未実行・空欄のまま**とし、実行後は
> この節と `docs/phase2.5-search-comparison.md` §9 へ実数値を記入する。
> 実行は commit の有無に関わらず、**事前登録文書の SHA-256 を job ログへ記録してから**
> 成果物を生成し、その job 証跡を回収する(N03)。

### 目的と背景

- **目的**: 探索の hash 非決定性(N01)を修正した後の公式再ゲート実行。
- **背景**: 修正前 digest(`b7fcc48b...`)で 2026-08-10 に実施した
  dev 3000–3784 / wallclock 100–299 / determinism 300–307 は、
  **修正前 digest の参考・採用結論の証拠としては再ゲートに不使用**とする。
- **事前登録時点では新範囲(4000–4784 / 400–599 / 600–607)を唯一の正式再ゲート成果物**
  としたが、a41 run 完走後の DeepSeek 監査により最終ゲート証拠にはしない。
  最終ゲート証拠は下記「Phase 2.5 真の最終 corrective 事前登録」の新範囲のみ
  (v4 の dev 5000–5784 / train 700–899 / 900–907 は cross audit で seed 衝突が判明し
  消費済み・最終証拠外)。

### 固定 protocol

| 項目 | 固定値 |
|---|---|
| dev 主判定 | `T=4500`、`H=2`、`SCALE=30`、`c=1.41`、各 **785ペア**、workers=96 |
| wall-clock 交差確認 | `T` cap **1,000,000**、`budget_seconds=2.0/決定点`、**200ペア**、workers=96 |
| parallel 決定論 | `T=4500`、**8ペア**、worker=1 vs 8 |
| 実行環境 | `/usr/bin/python3`(**3.9.x**)、**`PYTHONHASHSEED=0`**、4 thread env=1 |
| 対戦 | ① PIMC vs Heuristic ② IS-MCTS vs Heuristic ③ PIMC vs IS-MCTS |
| 判定 | Holm primary(①②、片側 α=0.05)+ secondary 両側 α=0.05 の state machine |
| 成功条件 | 既存「再検証の成功判定」1〜5 と同一(新 seed 範囲で充足) |

### 使用する seed 範囲(事前予約)

既存使用済み index を確認した上で、衝突しない未使用範囲を新規予約する:

| 実験 | 採用 N | 新 group / range | 状態 |
|---|---:|---|---|
| dev 主判定 | 785 | `dev` / **4000–4784** | **消費済み・最終ゲート証拠にしない(a41 run。下記「a41 run 実績」参照)** |
| wall-clock 交差確認 | 200 | `train` / **400–599** | **消費済み・最終ゲート証拠にしない(a41 run)** |
| parallel 決定論 | 8 | `train` / **600–607** | **消費済み・最終ゲート証拠にしない(a41 run)** |

理由:
- dev 3000–3784 は N01 修正前 digest で消費済みのため、修正後は 4000 以降を新規採用
- train 100–299 / 300–307 は N01 修正前 digest で消費済みのため、
  wallclock 400–599 と determinism 600–607 を新規採用
  (determinism は wallclock と**重複しない未使用範囲**に分離)

### 実行コマンド(固定)

```
# full dev compare
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --pairs 785 --budget 4500 \
    --horizon 2 --scale 30 --c 1.41 --workers 96 --start-index 4000 \
    --records docs/eval-runs/phase2.5-stage2d-dev-4000.jsonl --execute

# wall-clock cross-check
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --group train --pairs 200 \
    --budget 1000000 --budget-seconds 2.0 --horizon 2 --scale 30 --c 1.41 \
    --workers 96 --start-index 400 \
    --records docs/eval-runs/phase2.5-wallclock-train-400-599.jsonl --execute

# parallel determinism
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --group train --pairs 8 \
    --budget 4500 --horizon 2 --scale 30 --c 1.41 --workers 8 \
    --start-index 600 --check-determinism \
    --records docs/eval-runs/phase2.5-determinism-train-600-607.json --execute
```

判定は新形式 protocol を使用:
`--protocol phase2.5-dev` / `--protocol phase2.5-wallclock`。
上記3本はそのまま実行できる固定コマンドであり、すべて `PYTHONHASHSEED=0 /usr/bin/python3`
を明示している。

### a41 run 実績と失効(2026-08-10 完走・DeepSeek 監査後)

本節の事前登録どおり、a41 run は下記を完走した。ただし**最終ゲート証拠にはしない**ため、
勝率・CI 等の実数値は本台帳へ転記せず、保存物
(`docs/eval-runs/phase2.5-stage2d-dev-4000*` / `docs/eval-runs/phase2.5-wallclock-train-400-599*` /
`docs/eval-runs/phase2.5-determinism-train-600-607.json`)に残す。

| 項目 | a41 run 実績 |
|---|---|
| dev 主判定(4000–4784、3×785ペア) | 完走・IS-MCTS 採用(保存物参照。最終証拠外) |
| wall-clock 交差確認(400–599、200ペア) | 完走(保存物参照。最終証拠外) |
| parallel 決定論(600–607、worker=1 vs 8) | 一致実測(最終証拠外。source digest 変更後の再実測が必要) |
| DeepSeek 監査(完走後) | CI 再計算・label 左右整合・det report runtime 保存・sampler 入力/csi 修正が必要 → **source digest 変更** |
| seed 範囲 | dev 4000–4784 / train 400–599 / 600–607 は**消費済み**・再利用禁止 |

> 本 run の数値・run_id は最終ゲート証拠として引用しない。旧 digest と同様、
> `--legacy` で歴史再現する履歴成果物として扱う。

## 判定宣言: Phase 2.5 最終 corrective 事前登録(v4 run): 実績と失効(2026-08-10)

> **これは結果を見る前の宣言である。** a41 run は完走したが、DeepSeek 監査で
> source digest を変える必須修正(CI 再計算・label 左右整合・det report runtime 保存・
> sampler 入力/csi 修正)が判明したため、最終ゲート証拠は下記の新範囲・新 output 名の
> 実行のみとする。**結果欄はすべて未実行・空欄のまま**とし、実行後はこの節と
> `docs/phase2.5-search-comparison.md` §10 へ実数値を記入する。
> 実行は commit の有無に関わらず、**事前登録文書の SHA-256 を job ログへ記録してから**
> 成果物を生成し、その job 証跡を回収する(N03)。
> strict protocol(`--protocol phase2.5-dev` / `--protocol phase2.5-wallclock`)の適用は
> この最終 corrective 新rangeのみとする。旧成果物(pre-N01 の dev 3000–3784 /
> train 100–299 / 300–307、および a41 の dev 4000–4784 / train 400–599 / 600–607)は
> `--legacy` による歴史再現であり、現行 strict protocol で再検証済みとはしない。

> **⚠ 2026-08-10 完了後追記(v4 run の失効)**: 本節の3範囲(dev 5000–5784 /
> train 700–899 / 900–907)は事前登録どおり開始したが、起動後の cross audit で
> **900–902 が既存 RSS 計測(train 900–902)と衝突**していることが判明し、
> 全 job を TERM した。3範囲は**消費済み・最終ゲート証拠にしない**。
> 最終ゲート証拠は下記「Phase 2.5 真の最終 corrective 事前登録」の新範囲のみ。

### 固定 protocol

| 項目 | 固定値 |
|---|---|
| dev 主判定 | `T=4500`、`H=2`、`SCALE=30`、`c=1.41`、各 **785ペア**、workers=96 |
| wall-clock 交差確認 | `T` cap **1,000,000**、`budget_seconds=2.0/決定点`、**200ペア**、workers=96 |
| parallel 決定論 | `T=4500`、**8ペア**、worker=1 vs 8 |
| 実行環境 | `/usr/bin/python3`(**3.9.x**)、**`PYTHONHASHSEED=0`**、4 thread env=1 |
| 対戦 | ① PIMC vs Heuristic ② IS-MCTS vs Heuristic ③ PIMC vs IS-MCTS |
| 判定 | Holm primary(①②、片側 α=0.05)+ secondary 両側 α=0.05 の state machine |
| 成功条件 | 既存「再検証の成功判定」1〜5 と同一(seed 範囲を下記新 range に読み替え) |

### 使用する seed 範囲と output 名(事前予約)

a41 run の消費済み範囲(dev 4000–4784 / train 400–599 / 600–607)と衝突しない
新規未使用範囲として予約したが、開始後の cross audit で det 900–907 が既存 RSS 計測
(train 900–902)と衝突し、全 job を TERM した。以下3範囲は**消費済み**:

| 実験 | 採用 N | 新 group / range | output 名 | 状態 |
|---|---:|---|---|---|
| dev 主判定 | 785 | `dev` / **5000–5784** | `docs/eval-runs/phase2.5-stage2e-dev-5000.jsonl` | **消費済み・最終ゲート証拠にしない(v4 run TERM)** |
| wall-clock 交差確認 | 200 | `train` / **700–899** | `docs/eval-runs/phase2.5-wallclock-train-700-899.jsonl` | **消費済み・最終ゲート証拠にしない(v4 run TERM)** |
| parallel 決定論 | 8 | `train` / **900–907** | `docs/eval-runs/phase2.5-determinism-train-900-907.json` | **消費済み・最終ゲート証拠にしない(900–902 が既存 RSS 計測と衝突。v4 run TERM)** |

### 実行コマンド(固定)

```
# full dev compare
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --pairs 785 --budget 4500 \
    --horizon 2 --scale 30 --c 1.41 --workers 96 --start-index 5000 \
    --records docs/eval-runs/phase2.5-stage2e-dev-5000.jsonl --execute

# wall-clock cross-check
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --group train --pairs 200 \
    --budget 1000000 --budget-seconds 2.0 --horizon 2 --scale 30 --c 1.41 \
    --workers 96 --start-index 700 \
    --records docs/eval-runs/phase2.5-wallclock-train-700-899.jsonl --execute

# parallel determinism
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --group train --pairs 8 \
    --budget 4500 --horizon 2 --scale 30 --c 1.41 --workers 8 \
    --start-index 900 --check-determinism \
    --records docs/eval-runs/phase2.5-determinism-train-900-907.json --execute
```

判定は新形式 protocol を使用:
`--protocol phase2.5-dev` / `--protocol phase2.5-wallclock`。
上記3本はそのまま実行できる固定コマンドであり、すべて `PYTHONHASHSEED=0 /usr/bin/python3`
を明示している。`stage2` は `--start-index` 明示必須。

### v4 run 実績と失効(2026-08-10 開始→TERM)

事前登録どおり開始したが、起動後の cross audit で seed 衝突が判明し、全 job を
TERM した。**最終ゲート証拠にはしない**ため、勝率・CI 等の実数値は本台帳へ転記しない。

| 項目 | v4 run 実績 |
|---|---|
| dev 主判定(5000–5784、3×785ペア) | 開始したが全 job TERM。数値・run_id は最終証拠外 |
| wall-clock 交差確認(700–899、200ペア) | 開始したが全 job TERM。数値・run_id は最終証拠外 |
| parallel 決定論(900–907、8ペア) | **900–902 が既存 RSS 計測(train 900–902)と衝突**。全 job TERM |
| seed 範囲 | dev 5000–5784 / train 700–899 / 900–907 は**消費済み**・再利用禁止 |
| 最終ゲート証拠 | **不使用**。下記「真の最終 corrective 事前登録」の新範囲のみ |

## 判定宣言: Phase 2.5 真の最終 corrective 事前登録(2026-08-10)

> **これは結果を見る前の宣言である。** v4 run(dev 5000–5784 / train 700–899 /
> 900–907)は cross audit で 900–902 が既存 RSS 計測と衝突し全 job TERM・消費済みと
> なったため、最終ゲート証拠は下記の新範囲・新 output 名の実行のみとする。
> **結果欄はすべて未実行・空欄のまま**とし、実行後はこの節と
> `docs/phase2.5-search-comparison.md` §11 へ実数値を記入する。
> 実行は commit の有無に関わらず、**事前登録文書の SHA-256 を job ログへ記録してから**
> 成果物を生成し、その job 証跡を回収する(N03)。
> strict protocol(`--protocol phase2.5-dev` / `--protocol phase2.5-wallclock`)の適用は
> この真の最終新rangeのみとする。旧成果物(pre-N01 の dev 3000–3784 / train 100–299 /
> 300–307、a41 の dev 4000–4784 / train 400–599 / 600–607、v4 の dev 5000–5784 /
> train 700–899 / 900–907)は `--legacy` による歴史再現であり、現行 strict protocol
> で再検証済みとはしない。

### 固定 protocol

| 項目 | 固定値 |
|---|---|
| dev 主判定 | `T=4500`、`H=2`、`SCALE=30`、`c=1.41`、各 **785ペア**、workers=96 |
| wall-clock 交差確認 | `T` cap **1,000,000**、`budget_seconds=2.0/決定点`、**200ペア**、workers=96 |
| parallel 決定論 | `T=4500`、**8ペア**、worker=1 vs 8 |
| 実行環境 | `/usr/bin/python3`(**3.9.x**)、**`PYTHONHASHSEED=0`**、4 thread env=1 |
| 対戦 | ① PIMC vs Heuristic ② IS-MCTS vs Heuristic ③ PIMC vs IS-MCTS |
| 判定 | Holm primary(①②、片側 α=0.05)+ secondary 両側 α=0.05 の state machine |
| 成功条件 | 既存「再検証の成功判定」1〜5 と同一(seed 範囲を下記新 range に読み替え) |

### 使用する seed 範囲と output 名(事前予約・実行済み)

v4 の消費済み範囲(dev 5000–5784 / train 700–899 / 900–907)と衝突しない
新規未使用範囲を予約する:

| 実験 | 採用 N | 新 group / range | output 名 | 状態 |
|---|---:|---|---|---|
| dev 主判定 | 785 | `dev` / **6000–6784** | `docs/eval-runs/phase2.5-stage2f-dev-6000.jsonl` | **実行済み(2026-08-10)**・証跡: `docs/eval-runs/phase2.5-csi-jobs-20260811-formal/` |
| wall-clock 交差確認 | 200 | `train` / **1100–1299** | `docs/eval-runs/phase2.5-wallclock-train-1100-1299.jsonl` | **実行済み(2026-08-10)**・証跡: `docs/eval-runs/phase2.5-csi-jobs-20260811-formal/` |
| parallel 決定論 | 8 | `train` / **1400–1407** | `docs/eval-runs/phase2.5-determinism-train-1400-1407.json` | **実行済み(2026-08-10)**・証跡: `docs/eval-runs/phase2.5-csi-jobs-20260811-formal/` |

### 実行コマンド(固定)

```
# full dev compare
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --pairs 785 --budget 4500 \
    --horizon 2 --scale 30 --c 1.41 --workers 96 --start-index 6000 \
    --records docs/eval-runs/phase2.5-stage2f-dev-6000.jsonl --execute

# wall-clock cross-check
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --group train --pairs 200 \
    --budget 1000000 --budget-seconds 2.0 --horizon 2 --scale 30 --c 1.41 \
    --workers 96 --start-index 1100 \
    --records docs/eval-runs/phase2.5-wallclock-train-1100-1299.jsonl --execute

# parallel determinism
PYTHONHASHSEED=0 /usr/bin/python3 scripts/compare_search.py --stage 2 \
    --group train --pairs 8 \
    --budget 4500 --horizon 2 --scale 30 --c 1.41 --workers 8 \
    --start-index 1400 --check-determinism \
    --records docs/eval-runs/phase2.5-determinism-train-1400-1407.json --execute
```

判定は新形式 protocol を使用:
`--protocol phase2.5-dev` / `--protocol phase2.5-wallclock`。
上記3本はそのまま実行できる固定コマンドであり、すべて `PYTHONHASHSEED=0 /usr/bin/python3`
を明示している。`stage2` は `--start-index` 明示必須。

### 判定結果(2026-08-10 正式実行済み)

| 対戦 | dev pairs | dev range | 勝率 | 95%CI | 勝/分/敗 | 壁時計 |
|---|---:|---|---:|---:|---|---:|
| ① PIMC vs Heuristic | 785 | 6000–6784 | **0.9834** | [0.9771, 0.9892] | 1544/0/26 | 122.3 s |
| ② IS-MCTS vs Heuristic | 785 | 6000–6784 | **0.9662** | [0.9567, 0.9752] | 1517/0/53 | 167.1 s |
| ③ PIMC vs IS-MCTS | 785 | 6000–6784 | **0.3465** | [0.3236, 0.3694] | 544/0/1026 | 274.1 s |

```
primary p値: pimc=0.00010 ismcts=0.00010 (片側 α=0.05, Holm)
棄却: ['ismcts', 'pimc']
secondary(pimc vs ismcts) 片側 p: 順方向=1.00000 逆方向=0.00010 → 両側 p=0.00020 (α=0.05) 点推定=0.3465
→ 直接対決で有意差あり → 採用: ismcts
```

| 検査項目 | 結果 |
|---|---|
| wall-clock 交差確認(train 1100–1299, pairs=200, budget_seconds=2.0) | **正式実行済み**。① 0.9850 [0.9725, 0.9950] 394/0/6、② 0.9800 [0.9650, 0.9925] 392/0/8、③ 0.2650 [0.2250, 0.3075] 106/0/294。primary p=0.00010/0.00010(Holm)・secondary 両側 p=0.00020 → **採用: ismcts**。証跡: `docs/eval-runs/phase2.5-csi-jobs-20260811-formal/`(wall-1100 exit_code=0)、`docs/eval-runs/phase2.5-wallclock-train-1100-1299.jsonl`・同 `-decision.txt` |
| worker=1 vs 8 並列一致(train 1400–1407, pairs=8, T=4500) | **正式実行済み**。3 labels(pimc-vs-heuristic / ismcts-vs-heuristic / pimc-vs-ismcts)すべてで worker=1 と worker=8 の決定論的 field が一致、`all_labels_matched: true`。証跡: `docs/eval-runs/phase2.5-csi-jobs-20260811-formal/`(det-1400 exit_code=0)、`docs/eval-runs/phase2.5-determinism-train-1400-1407.json` |
| N03 事前登録文書 hash を job ログへ記録 | **記録済み**。`formal-chain/job.log` 冒頭(2026-08-10T16:36:29Z・host=csi-hpc-01・Python 3.9.21)に事前登録文書の SHA-256(eval-seeds.md `fabba4e1...` / phase2.5-search-comparison.md `26044473...`)と Python source digest `d1d127e4...` を記録 |

```
wall-clock primary p値: pimc=0.00010 ismcts=0.00010 (片側 α=0.05, Holm)
棄却: ['ismcts', 'pimc']
secondary(pimc vs ismcts) 片側 p: 順方向=1.00000 逆方向=0.00010 → 両側 p=0.00020 (α=0.05) 点推定=0.2650
→ 直接対決で有意差あり → 採用: ismcts
```

**正式実行の環境と証跡**:

- 実施日: **2026-08-10(UTC 16:36–17:12)**・host `csi-hpc-01`・`/usr/bin/python3` = **CPython 3.9.21**・
  `PYTHONHASHSEED=0`・thread env 4変数すべて `1`・workers=96(parallel 決定論のみ 8)。
- 3本は**逐次実行**した(wall-clock 実験の思考時間計測が他ジョブの負荷で歪むのを避けるため)。
- job 証跡: `docs/eval-runs/phase2.5-csi-jobs-20260811-formal/`。`formal-chain/exit_code` と
  `formal-results/{dev-6000,wall-1100,det-1400}.exit_code` はすべて **0**(実行スクリプト:
  `formal_chain.sh`)。
- 記録: `docs/eval-runs/phase2.5-stage2f-dev-6000.jsonl`・同 `-decision.txt`、
  `docs/eval-runs/phase2.5-wallclock-train-1100-1299.jsonl`・同 `-decision.txt`、
  `docs/eval-runs/phase2.5-determinism-train-1400-1407.json`。

---

## 判定宣言: Phase 3 — PPO学習パイプラインの健全性(2026-08-11 事前登録)

**実行前に登録する。** 実行後にペア数・閾値・対立仮説を変更しない。追加が必要になった場合は
未使用 seed 範囲で最初からやり直す(逐次多重検定の禁止、core-api §10.4)。

起点コミット: `1332ff1`(Phase 2.5 クローズ)。設計正本: `docs/phase3-design.md` v1。

### 使用済み index(2026-08-11 時点。再利用禁止)

| group | 使用済み範囲 |
|---|---|
| `dev` | 0–499 / 1000–1499 / 2000–2499 / 2500–2999 / 3000–3784 / 4000–4784 / 5000–5784 / 6000–6784 |
| `train` | 0–49 / 100–299 / 300–307 / 400–599 / 600–607 / 700–899 / 900–907 / 1100–1299 / 1400–1407 |
| `accept` | なし(Phase 6 の最終受け入れまで温存する) |

### Phase 3 の予約(新規・未使用を網羅確認済み)

| 用途 | group / 範囲 | ペア数 | 出力 |
|---|---|---:|---|
| 自己対戦(学習) | `train` / **2000 以降**(上限なし) | — | チェックポイントのみ。判定に使わない |
| **完了判定 vs ランダム** | `dev` / **7000–7799** | **800** | `docs/eval-runs/phase3-vsrandom-dev-7000-7799.jsonl` |
| 昇格判定 第1回 | `dev` / **7800–8599** | 800 | `docs/eval-runs/phase3-promote-dev-7800-8599.jsonl` |
| 固定ベンチマーク群 | `dev` / **9000–9799** | 800 | `docs/eval-runs/phase3-bench-dev-9000-9799.jsonl` |
| 昇格判定 第2回以降 | `dev` / 10000 以降を 800 刻み | 800 | 実施時に本台帳へ追記してから実行する |

### 完了判定の固定条件(PLAN.md Phase 3 完了条件)

| 項目 | 固定値 |
|---|---|
| 対象 | 学習済み `PolicyAgent`(greedy)vs `RandomAgent`、2人戦・席交換ペア |
| 閾値 | **0.50** |
| 対立仮説 `p_alt` / Δ | 0.55 / 5pt |
| σ上界 | 0.498 |
| 検出力 | 80%(計算値 777 → 採用 **800ペア**、core-api §10.4 の表) |
| seed group / index | `dev` / **7000–7799** |
| bootstrap | ペア単位クラスタ、B=10,000、seed=`2026072102`(Phase 2 以来の固定値) |
| 合格規則 | 95% CI 下限 **> 0.50**(**厳密不等号**。`>=` にしない) |
| 前提版 | `rules_version` 0.3.1 / `OBS_VERSION` 0.3.0 / `API_VERSION` 0.1.0 / `OBS_ENCODER_VERSION` 1.0.0 / `ACTION_ENCODER_VERSION` 1.0.0 / `REWARD_VERSION` 1.0.0 |

### 昇格・退行の判定条件(PLAN.md の「仮値」を Phase 3 で確定)

| 項目 | 固定値 |
|---|---|
| 昇格① | 現行チャンピオンとの直接対戦(dev 800ペア・席交換)で 95%CI 下限 **> 0.50**(厳密不等号) |
| 昇格② | 固定ベンチマーク群(`HeuristicAgent` + 指定過去世代)への勝率点推定が、チャンピオンの記録値 **− 3pt 以上** |
| 退行 | 固定ベンチマーク勝率がチャンピオン比 **5pt 超**低下 → ロールバック |
| 分離 | 学習用プール = `train` seed / 昇格・退行判定 = `dev` seed / 最終受け入れ = `accept` seed |

昇格は**①と②の両方**を満たすときだけ行う。判定中は相手・混合比率・対局 seed を固定する。

### 新しい seed ストリーム種別(`derive_seed`)

| 種別 | 用途 | `SEAT_INDEPENDENT_STREAMS` |
|---|---|---|
| `"pool"` | 相手プールの抽選 | 入れない |
| `"seat"` | 学習時の席ランダム化 | 入れない |
| `"ppo"` | ミニバッチのシャッフル・初期重み | 入れない |

既存の `"env"` / `"setup"` は席非依存のまま。**ペア評価では席ランダム化を使わず
`seat_variant ∈ {0,1}` で明示的に席を交換する**(席バイアスをペア内で相殺するため)。

### Sol 引き継ぎ①に伴う p 値の再計算について(予告)

`scripts/decide_search.py` の `one_sided_p` は H0 下へ再中心化されていない
(`docs/reviews/2026-08-11-sol-phase2.5-gate-3.md` [中])。Phase 3 の Wave A5 でこれを
H0 再中心化 bootstrap / permutation / exact 符号検定へ修正する。

**既存 `docs/eval-runs/*` は歴史的成果物として不変のまま保持する。** 修正後の実装で
保存済み `pair_scores` から再計算した値は、本台帳へ**追記**する(既存の記録を上書きしない)。
Opus が事前に独立計算した exact 符号検定は 直接対決 両側 p = **3.95e-33**(dev 6000–6784)/
**5.78e-20**(wall 1100–1299)であり、記録値 0.00020 は B=10,000 の分解能下限に張り付いた
保守値である。**Phase 2.5 の結論(SO-ISMCTS 採用)は変わらない。**

### Sol 引き継ぎ①の実施結果 — p 値の再計算(2026-08-11、Phase 3 Wave A5)

`scripts/decide_search.py` の `one_sided_p` を **exact 条件付き符号検定**へ差し替えた
(引き分け 0.5 ペアは非情報として n から除外。B=10,000 bootstrap の分解能下限 1e-4 は使わない)。

**保存済みデータは一切書き換えていない。**下表は同じ `docs/eval-runs/*.jsonl` を新実装で
読み直した**追記**であり、既存の記録を上書きするものではない。

| 判定 | 勝率(再現) | 主判定 p(Holm 前) | 直接対決 両側 p | 採用 |
|---|---|---|---|---|
| dev 6000–6784 | 0.9834 / 0.9662 / **0.3465** | pimc `3.298e-229` / ismcts `7.503e-217` | **`3.952e-33`** | ismcts |
| wall train 1100–1299 | 0.9850 / 0.9800 / **0.2650** | pimc `3.983e-59` / ismcts `1.593e-58` | **`5.775e-20`** | ismcts |

- **勝率・CI・勝/分/敗はすべて記録済みの値と完全一致**した(実装差し替えが対局結果に影響しないことの確認)。
- 直接対決の両側 p は、Opus が提出前監査で独立計算した exact 符号検定
  (`docs/reviews/2026-08-11-opus-phase2.5-pre-gate-audit.md` §4.1)の
  **dev 3.95e-33 / wall 5.78e-20 と一致**した。
- 旧記録値 `0.00020` は B=10,000 の分解能下限に張り付いた保守値であり、
  **Phase 2.5 の結論(SO-ISMCTS 採用)は変わらない。**
- 副次的に、`decide_search.py` 内の「有意だが `point_estimate=0.5`」分岐が
  exact 検定の下では到達不能(点推定ちょうど 0.5 は両側 p=1.0)であることが判明した。
  死にコードとして Phase 3 の残留リスクに記録する。

### Phase 3 完了判定の実行結果(2026-08-13、csi-enshu)— **合格**

事前登録どおり `dev` **7000–7799**(800ペア = 1600局)で実行した。

| 項目 | 値 |
|---|---|
| 対象 | `PolicyAgent`(gen546、greedy)vs `RandomAgent`、2人戦・席交換ペア |
| 勝 / 分 / 敗 | **1600 / 0 / 0** |
| 点推定 | **1.0000** |
| 95% CI | **[1.0000, 1.0000]** |
| 合否(CI下限 **> 0.50**、厳密不等号) | **合格** |
| bootstrap | ペア単位クラスタ、B=10,000、seed=`2026072102` |
| 対応表 | `docs/eval-runs/phase3-vsrandom-dev-7000-7799.jsonl`(1600行) |
| 対応表 SHA-256 | `622437f2a13977dfe5b360b78a0dddb5255751df30733ac1af1ba00e2e91f5a3`(local/remote 一致) |
| プロセス終了コード | **0** |

**学習ジョブ**(切断耐性、`scripts/csi_job.sh` 経由):

| 項目 | 値 |
|---|---|
| 構成 | `--workers 4 --episodes 64 --generations 0 --max-seconds 1800 --keep-latest 5 --start-index 2000` |
| 到達世代 | **546**(training_steps 546) |
| スループット | 3,143〜3,437 decision/s |
| entropy | 1.09 → **0.30**(方策が尖り、学習が進行) |
| 破棄局 | **0** |
| 終了 | 壁時計 deadline でチェックポイント保存後に正常終了、`exit_code=0` |
| 環境 | `csi-hpc-01` / CPython **3.9.21** / torch **2.8.0+cpu** / 4スレッド環境変数すべて 1 |
| 学習 seed | `train` 2000 以降のみ(`dev` / `accept` 不使用) |

**Opus が対応表だけから独立検証した内容**:

- 1600行、`group` は `dev` 単一、index 範囲 7000–7799
- `seat_variant` 分布 800/800、`target_seat` 分布 800/800
- **同一配札でないペア 0 件**、**席が交換されていないペア 0 件**
- `rules_version` 0.3.1 / `api_version` 0.1.0 / `obs_version` 0.3.0 /
  `obs_encoder_version` 1.0.0 / `action_encoder_version` 1.0.0 / `reward_version` 1.0.0 /
  `cards_sha256` `f1f201d4…` がいずれも**単一値**

実行前に job ログの冒頭へ事前登録文書の SHA-256 を記録した
(`eval-seeds.md` = `8d5a843e…`、`phase3-design.md` = `d274eba8…`、`PLAN.md` = `4047a5c5…`)。

### Phase 3B ゲート対応後の正式再実行(2026-08-13、csi-enshu)— **合格**

Phase 3B ゲート1回目の指摘([高]再現メタデータ / [高]昇格判定の凍結)に対応したうえで、
**学習と判定をやり直した**。本節の実行が Phase 3 の**正式証跡**である。

#### seed の再使用について(自己申告)

`dev` **7000–7799** は、指摘対応前の実行(§前節)で**一度消費している**。本再実行は
**同じモデル系列・同じ判定条件での証跡作り直し**であり、閾値・対立仮説・ペア数・
seed 範囲はいずれも変更していない(逐次多重検定に当たる「有意になるまで繰り返す」
性質のものではない)。ただし**同一 seed 範囲を2回使った事実は隠さず記録する**。

#### 学習ジョブ

| 項目 | 値 |
|---|---|
| job | `docs/eval-runs/phase3-csi-jobs-20260813/train-4w-30m/` |
| 構成 | `--workers 4 --episodes 64 --generations 0 --max-seconds 1800 --keep-latest 5 --start-index 2000` |
| 到達世代 | **532**(training_steps 532) |
| スループット | 約 3,400 decision/s |
| 破棄局 | **0** |
| 終了 | 壁時計 deadline でチェックポイント保存後に正常終了、**exit_code=0** |
| job.log 冒頭 | **事前登録文書の SHA-256 と git HEAD を実際に記録**(`eval-seeds.md` `0e3f2658…` / `phase3-design.md` `8bb4f3cc…` / `PLAN.md` `4047a5c5…`、git `f397e1cc…`、作業ツリー差分 **0行**) |
| 実効スレッド数 | **spawn worker 4つすべて `Threads=1`**(RSS 147〜154 MB)、親は 6(torch 内部管理)。4環境変数すべて 1 |

#### 完了判定

| 項目 | 値 |
|---|---|
| 対象 | `PolicyAgent`(gen532、greedy)vs `RandomAgent` |
| seed | `dev` **7000–7799**(800ペア = 1600局) |
| 勝 / 分 / 敗 | **1600 / 0 / 0** |
| 点推定 / 95%CI | **1.0000** / **[1.0000, 1.0000]** |
| 合否(CI下限 **> 0.50** 厳密不等号) | **合格** |
| 対応表 | `docs/eval-runs/phase3-vsrandom-dev-7000-7799-v3.jsonl`(1600行) |
| 終了コード | **0** |

#### 成果物の SHA-256(local / remote 一致を確認)

| ファイル | SHA-256 |
|---|---|
| `gen532.meta.json` | `9f674cc4a558cba597d7cb06d72353126475a5ea8b7dd28f9965eba0656f0bb1` |
| `gen532.weights.pt` | `b9b4d61efd13db51376b7251cbe6739cecd8d4d4dad0cf43de478ff6c6b895b1` |
| `gen532.optimizer.pt` | `12e3b6c30d0fe348658dedafacc51a7df7b55854d80a34def817e6358164c8b3` |
| 対応表 jsonl | `17436fe06dc1c5aaab7480dc7416475d9d00250e59b46bb7e036c11a62789c69` |

チェックポイント3成果物は `docs/eval-runs/phase3-final-checkpoint/` へ回収済み。

#### 再現メタデータ(gen532、実値)

```
git_commit             f397e1ccdbc1196200597d7fab81d7ceb85783d6   ← クリーンな sha(dirty なし)
generation 532 / training_steps 532 / root_seed 20260812
python_version 3.9.21 / torch_version 2.8.0+cpu / created_at 2026-08-13T12:35:26Z
rules_version 0.3.1 / cards_data_version 0.3.0 / obs_version 0.3.0
obs_encoder_version 1.0.0 / action_encoder_version 1.0.0 / reward_version 1.0.0
num_players 2 / model_arch {action_dim:109, action_hidden:[128], hidden:[256,256], obs_dim:569}
eval_results  [{opponent:random, group:dev, 7000-7799, 800ペア, 1600勝0分0敗,
                point_estimate:1.0, ci:[1.0,1.0], threshold:0.5, passed:true,
                records_sha256:17436fe0…, bootstrap B=10000 seed=2026072102}]
```

#### Opus が対応表だけから独立検証した内容

1600行 / 800ペア、`group` は `dev` 単一、index 7000–7799、
`seat_variant` 800/800、`target_seat` 800/800、
**同一配札でないペア 0 件**、**席が交換されていないペア 0 件**。

#### 実効スレッド数の実測(Rocky 実機、raw 証跡)

Phase 3B ゲート2回目の指摘(測定ファイルが証跡に無い)を受けて、**4-worker スモークを
再実行し `/proc/<pid>/status` の生出力を加工せず保存**した。

- 証跡: `docs/eval-runs/phase3-csi-jobs-20260813/thread-smoke/thread-measurement.txt`(80行)
- SHA-256: `5bcd50244bcd2f70f2f992626735d619bdeaa368aeee6757e5e175142f248f93`(local/remote 一致)
- host `csi-hpc-01` / uname `5.14.0-503.21.1.el9_5.x86_64` / 2026-08-13T12:48:09Z
- 構成: `--workers 4 --episodes 64`、job の `exit_code=0`

| プロセス | pid | Threads | VmRSS |
|---|---|---:|---:|
| 親(python) | 3406219 | **6**(torch 内部管理) | 926,284 kB |
| spawn worker | 3406602 | **1** | 129,024 kB |
| spawn worker | 3406610 | **1** | 141,312 kB |
| spawn worker | 3406618 | **1** | 129,024 kB |
| spawn worker | 3406621 | **1** | 125,952 kB |

**親・全 worker とも `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`NUMEXPR_NUM_THREADS` がすべて `1`** であることを `/proc/<pid>/environ` から確認した。

---

## 判定宣言: Phase 4 — vs ヒューリスティック(2026-08-14 事前登録)

Phase 4 の完了条件(PLAN.md §4)は **vs ヒューリスティック勝率 CI下限 80% 超**
(2人戦・評価seed固定・両席同数)+ Sol xhigh レビュー通過。本節はその完了判定の
**事前登録**である。実行後にペア数・閾値・対立仮説・seed 範囲を変更しない
(逐次多重検定の禁止、core-api §10.4)。

### Phase 4 の使用済み index(2026-08-14 時点。再利用禁止)

Phase 3 までの使用済み(0–499 / 1000–1499 / 2000–2499 / 2500–2999 / 3000–3784 /
4000–4784 / 5000–5784 / 6000–6784 / 7000–7799)に加え、Phase 4 では次を消費する。

| group | 範囲 | 用途 |
|---|---|---|
| `dev` | **9000–9799** | Phase 4 固定ベンチマーク群の確立(チャンピオン gen532 vs `HeuristicAgent`。昇格判定②のチャンピオン基準値) |

### Phase 4 の予約(新規・未使用)

| 用途 | group / 範囲 | ペア数 | 出力 |
|---|---|---:|---|
| **完了判定 vs ヒューリスティック** | `dev` / **10000–10799** | **800** | `docs/eval-runs/phase4-vsheuristic-dev-10000-10799.jsonl` |

`dev` 7800–8599(昇格判定 第1回)と `dev` 10000 以降の 800 刻み(昇格判定 第2回以降)は
Phase 3 で予約済みのまま。Phase 4 完了判定はこのうち未使用の **10000–10799** を
専用に充てる(昇格判定と完了判定の seed を分離し、判定規則を結果に応じて変える余地を
残さない)。

### 完了判定の固定条件(PLAN.md Phase 4 完了条件)

| 項目 | 固定値 |
|---|---|
| 対象 | 学習済み `PolicyAgent`(greedy)vs `HeuristicAgent`、2人戦・席交換ペア |
| 閾値 | **0.80** |
| 対立仮説 `p_alt` / Δ | 0.85 / 5pt(core-api §10.4 の Phase 4 表) |
| seed group / index | `dev` / **10000–10799** |
| ペア数 | **800**(=1600局。core-api §10.4 の 500ペア以上を満たす) |
| bootstrap | ペア単位クラスタ、B=10,000、seed=`2026072102`(Phase 2 以来の固定値) |
| 合格規則 | 95% CI 下限 **> 0.80**(**厳密不等号**。`>=` にしない) |
| 前提版 | 学習ジョブ実行時点の `rules_version` / `OBS_VERSION` / `API_VERSION` / encoder / reward 版を checkpoint メタへ記録済みであること |

### 実行コマンド(固定)

```sh
python scripts/eval_agent.py checkpoint:<champion>.meta.json heuristic \
  --group dev --start-index 10000 --pairs 800 --workers 4 \
  --records docs/eval-runs/phase4-vsheuristic-dev-10000-10799.jsonl --execute
```

### 固定ベンチマーク群の確立(2026-08-14 実行)— チャンピオン基準値

昇格判定②のチャンピオン基準値を確立した。対象は Phase 3 のチャンピオン gen532。

| 項目 | 値 |
|---|---|
| 対象 | `PolicyAgent`(gen532、greedy)vs `HeuristicAgent` |
| seed | `dev` **9000–9799**(800ペア = 1600局) |
| 勝 / 分 / 敗 | **1542 / 0 / 58** |
| 点推定 / 95%CI | **0.9637** / **[0.9544, 0.9725]** |
| 対応表 | `docs/eval-runs/phase4-bench-dev-9000-9799.jsonl`(1600行、SHA-256 `be35e037…`) |
| チャンピオン基準値 | `champion_benchmark.heuristic = 0.9637` |
| job | `~/mywork/jobs-phase4/bench-dev-9000-9799`(csi-enshu、exit_code=0) |

### 完了判定の実行結果(2026-08-14 実行)— **合格**

| 項目 | 値 |
|---|---|
| 対象 | `PolicyAgent`(gen532、greedy)vs `HeuristicAgent` |
| seed | `dev` **10000–10799**(800ペア = 1600局、事前登録どおり) |
| 勝 / 分 / 敗 | **1551 / 0 / 49** |
| 点推定 / 95%CI | **0.9694** / **[0.9606, 0.9775]** |
| 合否(CI下限 **> 0.80** 厳密不等号) | **合格**(0.9606 > 0.80) |
| 対応表 | `docs/eval-runs/phase4-vsheuristic-dev-10000-10799.jsonl`(1600行、SHA-256 `47cb6137…`) |
| job | `~/mywork/jobs-phase4/vsheuristic-dev-10000-10799`(csi-enshu、exit_code=0) |

実行時点の前提版は gen532 チェックポイントのメタに記録済み(`rules_version` 等)。
本判定は事前登録した seed 範囲・閾値・ペア数で1回だけ実行した。

### 昇格・退行判定の実運用(2026-08-14 実行)— **reject(gen532 がチャンピオンのまま)**

Phase 4 の本学習(gen532 を起点に自己対戦を 38 世代継続して gen570 まで到達)で得た
challenger gen570 に対して、昇格・退行判定を実運用した(残留リスク #1 の閉塞)。
チャンピオンは Phase 3 の gen532、チャンピオン基準値は本節で確立した
`champion_benchmark.heuristic = 0.9637`。

| 判定 | 値 |
|---|---|
| direct(challenger vs champion) | gen570 vs gen532、`dev` **7800–8599**、800ペア。**803/0/797**、点推定 0.5019、CI [0.4781, 0.5250] |
| 昇格①(CI下限 > 0.50) | **不成立**(0.4781 < 0.50) |
| benchmark(challenger vs heuristic) | gen570 vs heuristic、`dev` **9000–9799**、800ペア。**1546/0/54**、点推定 0.9663、CI [0.9569, 0.9750] |
| 昇格②(点推定 >= チャンピオン基準値 − 3pt = 0.9337) | **成立**(0.9663 >= 0.9337) |
| 退行(チャンピオン比 5pt 超低下 = < 0.9137) | **なし** |
| 判定 | **reject**(formal=true。gen570 は昇格せず、gen532 がチャンピオンのまま) |
| 対応表 | `phase4-promote-dev-7800-8599-direct.jsonl` / `-bench.jsonl`(各1600行) |
| 判定 JSON | `docs/eval-runs/phase4-promote-gen570.json`(frozen_models の SHA-256 含む) |
| job | `~/mywork/jobs-phase4/promote-gen570`(csi-enshu、exit_code=0) |

`dev` 7800–8599 は本判定で消費した。`dev` 9000–9799 は固定ベンチマークとして
チャンピオン基準値の確立と challenger 評価の両方に使う(固定ベンチマークの意図)。
次の昇格判定は `dev` 10800 以降の 800 刻みを使う。

**2026-08-16 追記(Phase 6 ゲート3回目の指摘)**: `scripts/promote.py` は正式条件の
`direct.start_index` を 7800 に固定しており、この台帳を守る限り**2回目の正式昇格が
実行できない**状態だった(消費済み帯を使うか台帳に違反するかの二択)。
`promote.py` を、7800 と 10800 以降の 800 刻み——**この台帳が予約した並び**——だけを
受け付ける形に直した。予約列に無い index は未使用でも拒否する。
ペア数・閾値・許容差・ベンチマーク帯は凍結したままである。
**使う帯は実行前に本台帳の予約表へ事前登録すること**(用途・帯・ペア数・合格条件)。**`CONSUMED-DEV-BANDS` ブロックへの追記は実行後**である(`promote.py` はブロックから「次の未消費帯」を計算するため、実行前に追記すると期待帯が 1 つ先へ進み、その帯での実行が拒否される。2026-09-05 に Sol ゲートの指摘で明確化)。

### 本学習と中断・再開の実績(2026-08-14、csi-enshu)

| 項目 | 値 |
|---|---|
| 起点 | gen532(Phase 3 チャンピオン、`next_episode_index=36048`)を `runs/phase4` へ複製して継続 |
| 初回ジョブ | `train-4w`。`--workers 4 --episodes 64 --generations 0 --max-seconds 1200 --keep-latest 5`。gen532 → gen555 まで継続 |
| 安全な中断 | ロールアウト中に SIGTERM(`csi_job.sh stop`)→「SIGTERM during rollout: checkpoint saved (generation=555)」→ **exit_code=0** |
| 再開ジョブ | `train-4w-resume`。`--generations 15`。「resumed from generation 555 / optimizer state restored」→ gen556 → gen570、**exit_code=0** |
| 残留プロセス | 0(`train_ppo.py` の孤立プロセスなし。flock はプロセス死亡でカーネル解放) |
| 到達モデル | gen570(git_commit `5d5d3c7…` / rules 0.3.1 / torch 2.8.0+cpu / root_seed 20260812 をメタへ記録) |

---

## seed 予約: Phase 5 — 探索あり vs なし(2026-08-14 予約)

Phase 5 の完了条件(PLAN.md §4)は **探索あり vs なしの同一配札ペア対局で勝率CI下限 > 50%**
+ Sol xhigh レビュー通過。本節は **index 範囲の予約**であり、判定条件の一部
(採用予算 `T`、チャンピオン、評価器構成)は M3 の較正で確定する。
**確定させてから、結果を1行も見る前に、下表を確定版へ書き換えて実行すること。**

### Phase 4 までの使用済み index(2026-08-14 時点。再利用禁止)

| group | 使用済み |
|---|---|
| `dev` | 0–499 / 1000–1499 / 2000–2499 / 2500–2999 / 3000–3784 / 4000–4784 / 5000–5784 / 6000–6784 / 7000–7799 / 7800–8599 / 9000–9799 / 10000–10799 |
| `train` | 0–49 / 100–299 / 300–307 / 400–599 / 600–607 / 700–899 / 900–907 / 1100–1299 / 1400–1407 / **2000 以降**(Phase 3・4 の自己対戦。gen532 時点で `next_episode_index=36048`) |
| `accept` | なし(Phase 6 の最終受け入れまで温存) |

`dev` **10800 以降の 800 刻み**は Phase 4 で「昇格判定 第2回以降」として予約済みである。
Phase 5 はこの帯を避け、20000 番台を使う。

### Phase 5 の予約(新規・未使用)

| 用途 | group / 範囲 | ペア数 | 出力 |
|---|---|---:|---|
| **完了判定 探索あり vs なし** | `dev` / **20000–20799** | **800** | `docs/eval-runs/phase5-search-dev-20000-20799.jsonl` |
| 予備(再判定が必要になった場合。改めて事前登録してから使う) | `dev` / **20800–21599** | 800 | — |
| 予算較正・思考時間計測・アブレーション | `train` / **500000–500999** | 最大500 | `docs/eval-runs/phase5-calibration-*.jsonl` |

`GROUPS` は `train` / `dev` / `accept` に固定されている(`naeco/engine/state.py:309`)。
Phase 5 のために新しい group を追加しない。**調整はすべて `train` 群で行い、
`dev` 20000–20799 は1回だけ実行する。**

### 完了判定の固定条件(実行前に確定させる)

| 項目 | 値 |
|---|---|
| 対象 | 探索あり(SO-ISMCTS + 学習済み policy/value)vs 探索なし(同一ネットの `PolicyAgent` greedy)、2人戦・席交換ペア |
| モデル | **gen532**(Phase 4 チャンピオン。gen570 は reject 済みなので使わない) |
| 閾値 | **0.50** |
| 対立仮説 `p_alt` / Δ | 0.55 / 5pt(core-api §10.4 の Phase 5 行) |
| seed group / index | `dev` / **20800–21599**(予備帯を最終試行として使用。20000–20799 は L1/S1 で消費済み) |
| ペア数 | **800**(=1600局) |
| bootstrap | ペア単位クラスタ、B=10,000、seed=`2026072102`(Phase 2 以来の固定値) |
| 合格規則 | 95% CI 下限 **> 0.50**(**厳密不等号**) |
| 探索予算 `T` | **4500**(1決定点あたりの状態遷移数)。M1 スパイクで net 呼出は L1 ≈40ms / L2 ≈270-530ms の追加に留まり、M2 gen532 スモークで平均 **477ms** / p95 **634ms** を実測。平均2秒/p95 5秒に十分余裕があるため据置(証跡 `docs/eval-runs/phase5-m1-spike.md`) |
| 探索構成 | **S2 + L1**(Phase 5 M4 反復で確定)。選択は **PUCT**(`c_puct=1.5`、`c=1.41` から分離・不流用)、prior は policy logits の softmax、**未展開行動も含めて PUCT で選択**(`unexpanded[0]` 強制展開は廃止)、FPU=0.5。葉は L1(ヒューリスティック rollout H=2 維持 + 葉で net value)。ルート Dirichlet ノイズなし。探索あり = `IsmctsAgent(..., c_puct=1.5, evaluator=NetEvaluator(gen532), policy_agent=PolicyAgent(gen532, greedy))`、探索なし = `PolicyAgent(gen532, greedy)` |
| 思考時間 | 平均2秒 / p95 5秒以内。基準機を明記し、ローカルと csi-enshu の両方を報告する |
| ハーネス | `scripts/compare_search_net.py`(label `searchon-vs-searchoff`)。`--checkpoint-dir docs/eval-runs/phase3-final-checkpoint --group dev --start-index 20800 --pairs 800 --budget 4500 --workers N --execute` |

### M4 反復の自己申告(value 非ゼロ和と train 選定、2026-08-14)

**value ヘッドはゼロ和でない**(M4 後実測、gen532 局面 n=1440): `u0+u1 平均=0.6888`
(ゼロ和なら 0.0)。両席に等しい楽観バイアス(≈+0.34)が乗るが、**手番バイアス
`(u0-u1)/2=-0.0052` はほぼ 0**。この共通バイアスは同一ノード内の行動間比較(argmax)で
ほぼ相殺されるため、S2+L1 の選択へ与える影響は小さい。反対称化 `p=0.5+(u_a-u_b)/4` は
候補として残すが、完了判定の必須条件とはしない。この点は Sol ゲートへ自己申告する。

**L1/S1 → S2 の変更根拠(train 選定)**:

| 構成 | 結果(対局ハーネス、gen532 greedy 相手) | 判定 |
|---|---|---|
| L1/S1(value-only UCB、prior 不使用) | dev 20000–20799 800ペアで **0.2725** CI [0.25125,0.294375] | 不通過 |
| **S2+L1(PUCT + softmax prior)** | train 500000–500029 30ペア **0.767** CI [0.667,0.867] / train 500030–500179 150ペア **0.717** CI [0.673,0.760] | **採択** |

S2 の prior が効くための前提(`unexpanded[0]` 強制展開の廃止)を修正した後の実測で、
探索ありが greedy に有意に勝つことを train 較正帯で確認した。S2 構成の決定論
(worker=1 vs 4 + replay-audit)も train 500500–500503 で一致を確認。

### 計算予算の扱い(事前に明記する)

同一計算予算の規則(PLAN.md §5 / core-api §10.5)は **探索方式どうしの比較**
(PIMC vs IS-MCTS、Phase 2.5)に適用される。Phase 5 の完了判定は
**モデルを固定し決定手続きだけを変える**比較であり、追加計算を投じる価値があるかを問う設計である。
したがって予算は揃えない。代わりに両腕の **遷移数・net 呼出数・平均/p95 思考時間**を併記する。

### 禁止事項の再確認

- 効果が +5pt 未満でも**ペア数を増やして有意になるまで繰り返さない**(core-api §10.4)。
  「効果量が 5pt 未満である」ことを結論として記録し、探索の採否をコストと併せて判断する。
- 結果を見てからペア数・閾値・対立仮説・seed 範囲・予算 `T` を変更しない。
  変更が必要になったら**未使用範囲で最初からやり直す**。

---

## Phase 6(WebUI・引き渡し)の seed 宣言

### 使用済み index(2026-08-14 時点。再利用禁止)

| group | 使用済み |
|---|---|
| `dev` | 0–499 / 1000–1499 / 2000–2499 / 2500–2999 / 3000–3784 / 4000–4784 / 5000–5784 / 6000–6784 / 7000–7799 / 7800–8599 / 9000–9799 / 10000–10799 / **20000–20799**(Phase 5 L1/S1 不合格判定)/ **20800–21599**(Phase 5 S2+L1 合格判定) |
| `train` | 0–49 / 100–299 / 300–307 / 400–599 / 600–607 / 700–899 / 900–907 / 1100–1299 / 1400–1407 / 2000 以降(自己対戦)/ 500000–500179(Phase 5 較正)/ 500500–500503(Phase 5 決定論) |
| `accept` | **0–799(Phase 8 最終受け入れ、2026-09-05)**。それ以前は未使用 |

### 機械可読な消費済み `dev` 帯(`scripts/promote.py` が読む正本)

上の表は**その時点の記録**なので、Phase ごとに何枚も残っている。機械が
「次の未消費帯」を決めるには、現在値がどれか一意に分かる必要がある。
そこで現在値だけをこのブロックに置く。**このブロックへの追記は昇格判定の実行後に行う**
(実行前に追記すると `promote.py` の期待帯が次へ進み、当該帯での実行が拒否される)。実行前に固定するのは上の予約表への事前登録である。 `promote.py` はこのブロックを読み、
**次の未消費の予約帯**と一致しない `direct.start_index` を正式条件として拒否する
(帯の飛ばし・再利用・複数帯からの事後選択を防ぐ。Sol ゲート4回目の指摘)。

ブロックが無い、または解析できない場合、`promote.py` は**実行を拒否する**
(fail-closed)。黙って既定値へ落ちない。

<!-- CONSUMED-DEV-BANDS:BEGIN -->
0-499 1000-1499 2000-2499 2500-2999 3000-3784 4000-4784 5000-5784 6000-6784
7000-7799 7800-8599 9000-9799 10000-10799 20000-20799 20800-21599
10800-11599 11600-12399 12400-13199 13200-13999 14000-14799
<!-- CONSUMED-DEV-BANDS:END -->

**Phase 7 の正式判定結果(2026-09-04 実行、証跡 `docs/eval-runs/phase7-step7/`)**

| 判定 | 帯 | 結果 | 合否 |
|---|---|---|---|
| 昇格(gen1300 vs gen532 greedy) | dev 10800–11599 | 0.9756 [0.9675, 0.9831]、1561/0/39。固定ベンチ heuristic 0.9981 | **promote** |
| G1(gen1300 greedy vs gen532+探索@4500) | dev 11600–12399 | 0.9712 [0.9625, 0.9794]、1554/0/46 | **合格**(CI 下限 > 0.50) |
| G2(gen1300+探索@1500 vs gen532+探索@4500) | dev 12400–13199 | 0.9812 [0.9744, 0.9875]、1570/0/30 | **合格**(CI 下限 > 0.65) |

次の昇格判定に使える帯は **dev 13200–13999**。

### Phase 7 G1/G2 の再実行(Sol ゲート 1 回目の指摘。2026-09-05 事前登録、各 1 回だけ実行)

Sol ゲート 1 回目(`docs/reviews/2026-09-05-phase7-sol-gate-1.md`)は、G1/G2 の出力にモデル SHA-256・HEAD・
台帳 SHA・時刻・`formal` が無く、`ladder.py` が既存出力を上書きするため「固定 gen1300 の一回限り判定」と
認定できないと裁定した。`ladder.py` に来歴(実行前のモデル凍結と実行後の再検査、既存出力拒否)を追加し、
**未使用の dev 帯で G1/G2 を各 1 回だけ再実行する**。dev 11600–12399 / 12400–13199 の結果は
「来歴の無い先行実行」として保持し、判定には使わない。

| 判定 | 左(挑戦者。固定済み gen1300、SHA は `phase7-step7/challenger.sha256`) | 右 | group / 範囲 | ペア数 | 合格条件 |
|---|---|---|---|---|---|
| G1(再) | gen1300 greedy | gen532+探索@4500 | `dev` / **13200–13999** | 800 | CI 下限 > 0.50 |
| G2(再) | gen1300+探索@1500 | gen532+探索@4500 | `dev` / **14000–14799** | 800 | CI 下限 > 0.65 |

実行後、これらの帯を CONSUMED-DEV-BANDS に追記した(2026-09-05)。以後の昇格判定の次帯は **14800–15599**。
結果は `docs/eval-runs/phase7-step7b/`(来歴つき)と `docs/phase7-design.md` §3.5l。

### G3(value 較正)の waiver(ユーザーの明示的な決定、2026-09-05)

Sol ゲート 1 回目は「G3 を達成するか、未達のまま Phase 7 を閉じるユーザーの明示的 waiver を記録する」ことを
求めた。ユーザーに「waiver を出す」か「G3 を達成してから閉じる」かを選択肢として提示し、
**ユーザーは「waiver を出す」を選択した**(本セッションの質問への回答。2026-09-05)。
したがって G3(ECE 半減、ゼロ和ずれ 1/3)は Phase 7 の完了条件から外し、実測値
(ECE 0.064→0.041、`mean(u0+u1)` 0.544→0.431。目標 0.032 / 0.181)を記録したうえで次 Phase の課題として引き継ぐ。
PLAN v2.4 の Phase 7 節の「G3 は完了条件に含めない」はこの決定を反映したもので、
その記述が結果コミットの後に書かれた事実(Sol の指摘)はここに認める。

(旧記述の削除: 2026-09-05。10800–11599 は Phase 7 の昇格判定で消費済み。次帯は上記のとおり 13200–13999。
予約列は 7800、次いで 10800 から 800 刻み。)

### Phase 7 段1 の診断帯(2026-09-03 事前登録。判定には使わない)

| 用途 | group / 範囲 | 規模 | 備考 |
|---|---|---|---|
| Elo ラダー(random / heuristic / gen532 greedy / gen532+探索の総当たり) | `train` / **501000–501099** | 100 ペア × 6 ペアリング | `scripts/ladder.py`。全ペアリングが同じ配札 |
| value 較正診断(gen532 greedy 自己対戦、両席の value を記録) | `train` / **502000–502399** | 400 局 | `scripts/value_calibration.py` |
| 予備(スクリプト修正後の再実行) | `train` / 501100–501199 / 502400–502799 | — | 501100–501119 を段2 の Gumbel スモーク(20 ペア)と exit000 スモークに使用 |
| Phase 7 自己対戦(Expert Iteration のデータ収集) | `train` / **600000 以降**(周ごとに 20000 刻み) | — | `scripts/selfplay_exit.py`。r1 = 600000–619999(gen532 教師、T=1500、p_full 0.5、両席) |
| Phase 7 の学習中ラダー | `train` / 501200–501999 | — | 昇格判定ではない。世代間の比較にだけ使う。使用済み: 501200–501299(r1)/ 501300–501399(r1b)/ 501400–501499(gen772)/ 501500–501599(big gen100)/ 501600–501699(r2)/ 501700–501799(gen955)/ 501800–501899(gen1000)/ 501900–501999(gen1100) |
| Phase 7 の学習中ラダー(追加帯。2026-09-05 に事後登録。判定ではない) | `train` / 505000–505099(gen1175)/ 505100–505299(plateau、200 ペア) | — | 事前登録漏れ。証跡 `docs/eval-runs/phase7-step3/ladder-1175.json` / `ladder-1300.json` |
| value 較正の追加(2026-09-05 に事後登録) | `train` / 502400–502799(gen635、予備帯を使用)/ 502800–503199(gen1300) | 各 400 局 | 証跡 `phase7-step4/calib-gen635.json` / `phase7-step7/calib-gen1300.json` |
| Phase 5 認定 bundle の再現(判定ではない) | `dev` / 20800–20819(消費済み帯の**再実行**、20 ペア) | — | 探索コード変更の等価性検証。結果・得点・決定論的メトリクスが保存済み v3 と全 40 局一致(`phase7-step2/equiv-phase5/`) |
| Phase 7 PPO 対照群(gen532 から再開) | `train` / 36048 以降の連番(gen532 の `next_episode_index` を継承) | — | `~/mywork/runs/phase7-ppo` |
| **新規学習の開始帯**(README の `--start-index` の値) | `train` / **900000 以降** | — | 2026-09-05 登録。Phase 7 の PPO 対照群は gen1300 で `next_episode_index=821456` に到達(36048〜821455 を消費)。新しい系統は 900000 から始め、使った範囲をここへ追記する |
| Phase 7 PPO 大型ネット(from scratch) | `train` / **200000 以降**の連番 | — | `~/mywork/runs/phase7-ppo-big`。Phase 3/4 が消費した 2000–36047 と重ねない。**注**: 対照群の連番が世代 700 付近で 200000 を超え、以後は両ランが同じ配札を別々に使っている(判定には無関係。自己対戦の独立性にだけ関わる) |

`train` 群なので判定の独立性には影響しないが、自己対戦の消費帯(2000 以降の連番)と
Phase 5 較正帯(500000–500999)から離して衝突を避ける。

### Phase 7 正式判定の事前登録(2026-09-04。実行前に登録。各 1 回だけ実行)

| 判定 | 左(挑戦者) | 右 | group / 範囲 | ペア数 | 合格条件 | ツール |
|---|---|---|---|---|---|---|
| 昇格(正式) | PPO 最終世代 greedy | gen532 greedy(現チャンピオン) | `dev` / **10800–11599**(台帳の次帯) | 800 | CI 下限 > 0.50 かつ固定ベンチ(dev 9000–9799、heuristic 0.9637 − 3pt) | `scripts/promote.py` |
| G1 | 同上 greedy | gen532+探索@4500(現在の「強」) | `dev` / **11600–12399** | 800 | CI 下限 > 0.50 | `scripts/ladder.py`(2 エージェント) |
| G2 | 同上+探索@1500 | gen532+探索@4500 | `dev` / **12400–13199** | 800 | CI 下限 > 0.65 | 同上 |

挑戦者は「plateau 確認後の最終世代」を実行前にここへ追記して固定する。
**挑戦者 = `phase7-ppo/gen1300`。固定の時刻は csi のコミット `d34c584`(2026-09-04 20:06:30 JST)で、
この行を含む台帳をコミットしてから 3 判定を起動した(起動 20:06:35、`promote` 完了 20:06:49、
G1 完了 20:10:00 頃、G2 完了 20:12:00 頃。`docs/eval-runs/phase7-step7/head` = d34c584)。**
設計書 §3.5i に当初「20:10 固定」と書いたのは執筆時刻の誤記で、訂正済み。
`promote.py` が出力する `seed_ledger.sha256` は実行後に計算されるため実行前の台帳状態の証明にはならず、
実行前の状態はコミット d34c584 の内容で示す。
plateau の根拠(`train` 505100–505299、200 ペア、`docs/eval-runs/phase7-step3/ladder-1300.json`):
gen1100 vs 1200 = 0.435、gen1100 vs 1250 = 0.420、gen1100 vs 1300 = 0.4725、gen1200 vs 1250 = 0.500、
gen1200 vs 1300 = 0.4925、gen1250 vs 1300 = 0.505。100 世代で差が CI 内に収まったので plateau と判定した。
3 判定は同時に(別帯で)実行した。
G1/G2 は `promote.py` の正式条件(checkpoint 同士)に収まらないため `ladder.py` で行うが、
同一配札・席交換・B=10000 のペア bootstrap は `harness` の同じ関数を使う。
ペア数 800 は core-api §10.4 の Phase 5 行(閾値 50% / 対立 55%)に合わせる。
G2 の閾値 0.65 は core-api の表に無い(Phase 7 設計 §0 で置いた目標値)。

### Phase 8 最終受け入れ(`accept` の初消費。2026-09-05 事前登録、1 回だけ実行)

方策を gen532 から gen1300 へ変えるので、`accept` seed を初めて消費する(W-4 の保護を解く条件を満たした)。
WebUI の「強」に載せる構成そのもの(gen1300 + SO-ISMCTS、PUCT c_puct=1.5、H=2、T=4500)を、
Phase 6 までの「強」(gen532 + 同構成)と同一配札・席交換で比べる。

| 判定 | 左(新「強」) | 右(旧「強」) | group / 範囲 | ペア数 | 合格条件 | ツール |
|---|---|---|---|---|---|---|
| 最終受け入れ | gen1300+探索@4500 | gen532+探索@4500 | `accept` / **0–799** | 800 | CI 下限 > 0.65(Phase 7 G2 と同じ閾値) | `scripts/ladder.py`(来歴つき) |

挑戦者モデルは `docs/eval-runs/phase7-champion/gen1300.{meta.json,weights.pt}`(SHA は `phase7-step7b/challenger.sha256`)。
`accept` はこれ以外に消費しない。結果は `docs/eval-runs/phase8/`。

**結果(2026-09-05 実行)**: 0.984375 [0.978125, 0.990000]、1575/0/25 → **合格**。`accept` 0–799 は消費済み。以後の最終受け入れは 800 以降を事前登録して使う。

### `accept` を Phase 6 で消費しないという判断(W-4)

**Phase 6 では `accept` seed を1つも消費しない。**

理由: PLAN.md §5 の設計上、`accept` は「**再調整が入ったとき**に、検証セットへの過適合を
避けて最終判定をやり直す」ための最終受け入れ専用セットである。**Phase 6 は方策を一切変更しない**
(チャンピオン gen532 をそのまま WebUI に載せるだけで、学習も探索構成の変更も行わない)。
したがって強さに関する新しい統計判定には情報量がなく、**Phase 5 の完了判定
(探索あり vs greedy = 0.738125、95%CI [0.719375, 0.75625]、dev 20800–21599、800ペア)が
そのまま Phase 6 の強さの根拠**になる。

PLAN.md Phase 6 の完了条件も「Sol xhigh レビュー通過 + ユーザーが『強』と対局して受け入れ」で
あって、CI 要件を含まない。台帳側の「Phase 6 の最終受け入れまで温存する」という記述は
**「Phase 6 で使う」という予約ではなく「それまで手を付けない」という保護**であり、
本節をもってその保護を Phase 6 以降へ持ち越す。

`accept` を使うべき将来の条件: **方策・観測・報酬・探索構成のいずれかを変更した**うえで
最終的な受け入れ判定を行うとき。そのときは使用範囲を実行前に本台帳へ事前登録し、1回だけ実行する。

### Phase 6 で seed を使う場所(判定ではない)

WebUI の検証は**統計判定ではない**ので、判定用の3群とは別枠として扱う。

- Playwright の不変条件アサート・ファズ・艦隊の探索的対局は、**環境 seed を固定した再現用**に
  使うだけであり、勝率の主張には一切使わない。
- WebUI は `game.new_game(2, seed, cards)` に**生の環境 seed** を渡す。判定帯は
  `eval_seed(group, index)`(= SHA-256 of `"naeco/v1/<group>/<index>"`)から
  `derive_seed(root, "env", index)` を経て作られるので、**名前空間が別**であり、
  生 seed `0`〜`99` はどの群のどの index とも衝突しない(`naeco/engine/state.py:301-316`、
  `naeco/ai/eval/harness.py:178-181`)。
- したがって WebUI 用の生 seed は**台帳の「使用済み」には計上しない**。判定に使っていないため。
  混乱を避けるため、その事実をここに明記しておく。
- 記録(Phase 9 A-C2、2026-10-09): `scripts/hint_time.py` の A5 計測は WebUI 生 seed **900000–900005** を使う(評価群の外・判定ではない・台帳の使用済みに計上しない。結果は `docs/eval-runs/phase9/hint-time-local.json`)。

---

## Phase 10 — 3・4 席の n 局巡回クラスタ評価(D4)の seed 台帳(2026-10-09、C3)

設計の正本は `docs/phase10-design.md` D4、プロトコルは core-api §10.6。2 人戦の台帳(上の各節と
CONSUMED-DEV-BANDS ブロック)とは**別欄**で管理し、2 人戦の欄には書き込まない。
`scripts/eval_agent.py --num-players 3|4` が下の 2 つの機械可読欄を読み書きする(手で編集しない)。

- **予約欄**(実行前登録): 1 行 1 予約の JSON。`id`・人数・group・帯(lo–hi)・用途・事前固定した対戦列
  `matchups`(正準 agent リスト。先頭が評価対象)・閾値・モデル SHA・検出力根拠 `power`・登録時刻。
  `matchups` が空の行は **hold**(帯だけを押さえ、対戦とモデルは後で凍結する)で、起動できない。
  hold は同じ id・人数・group・帯・用途・閾値のまま対戦を確定する形でだけ置き換えられる。
- **消費欄**: 起動時(対局開始前)に 1 対戦 1 行を追記する。**起動した帯は失敗・中断でも消費扱い**。
  同じ run_id の欠けた局だけを `--resume` で決定論的に再開でき、全局が揃うまで合否を出さない。
- 登録・起動は次のいずれかと重なれば拒否する(fail-closed): 2 人戦 CONSUMED-DEV-BANDS、
  2 人昇格の予約列(dev 7800–8599 / 固定ベンチ dev 9000–9799 / **dev 10800 以降の 800 刻み、上限なし**)、
  accept 0–799(Phase 8 消費)、既存の多人数予約・消費。同帯は事前固定した対戦の間でだけ共用できる。
- **多人数専用 dev 領域 30000–39999**(2026-10-09 オーケストレータ裁定、t_da064f08。phase10-design D4 改訂行):
  用途は 3・4 席の初期固定ベンチと昇格判定だけ。2 人昇格の予約列(dev 10800 以降 800 刻み)とは**独立**で、
  2 人列は無変更・上限なし(`promote._next_reserved_direct_band` は 14800 のまま)。2 人側はこの領域を使わない。
  ツールは dev 10800 以降のうちこの領域だけを 2 人列の拒否から外す(`eval_agent.MULTI_DEV_REGION`)。
  割当: 初期固定ベンチ 3 席 30000–30799 / 4 席 30800–31599、昇格 1 回目 3 席 31600–32399 / 4 席 32400–33199。
  以後の昇格は同領域内で連続未使用の 800 帯を順に割り当てる。Phase 5 の 20000 番台とは重ならない。
- formal(dev / accept)は 800 クラスタ固定。合否は 95% 両側 percentile クラスタ bootstrap
  (B=10,000、seed=2026072102)の CI 下限 `>` 閾値(厳密不等号)。勝利指標は `ranking[target_seat] == 0`。

### D4 の固定条件と検出力(実行前登録。結果を見てから変えない)

| 項目 | 3 席 | 4 席 |
|---|---:|---:|
| 完了閾値(final accept) | **0.60** | **0.50** |
| 対立仮説(閾値 + 0.05) | 0.65 | 0.55 |
| 保守 sigma(クラスタ値は 0..1) | 0.5 | 0.5 |
| 必要クラスタ `ceil(((1.96+0.842)*0.5/0.05)^2)` | 786 | 786 |
| 採用クラスタ / 局 | **800 / 2400** | **800 / 3200** |
| 正規近似の検出力 `Phi(0.05*sqrt(800)/0.5 - 1.96)` | 0.8074 | 0.8074 |
| 片側 alpha(95% 両側 CI 下限) | 0.025 | 0.025 |
| Bonferroni(3 席・4 席の 2 判定) | family alpha <= 0.05 | 同左 |

検出力は正規近似であり、bootstrap の有限標本保証とは主張しない。値は
`naeco.ai.eval.harness.d4_power_record` が計算し、各予約行の `power` に保存する。

### 設計案の帯の再検査結果(2026-10-09)

| 設計案(phase10-design D4) | 再検査 | 登録 |
|---|---|---|
| final accept 800–1599(3 席) | 重なりなし | **hold 登録**(`p10-accept-3p`、閾値 0.60) |
| final accept 1600–2399(4 席) | 重なりなし | **hold 登録**(`p10-accept-4p`、閾値 0.50) |
| (旧案)初期固定ベンチ dev 14800–15599 / 15600–16399 | **重複**: 14800–15599 は本台帳が「以後の昇格判定の次帯」と明記した 2 人昇格の次帯。どちらも「dev 10800 以降の 800 刻み」(上限なし)の列内 | **未登録**(拒否。下の多人数専用領域で置換) |
| (旧案)昇格 1 回目 dev 16400–17199 / 17200–17999 | **重複**: 同じ 2 人昇格の予約列内 | **未登録**(拒否。同上) |
| 初期固定ベンチ dev 30000–30799(3 席) | 多人数専用領域内・重なりなし | **hold 登録**(`p10-bench-3p`、閾値なし。判定は champion − 3pt の比較) |
| 初期固定ベンチ dev 30800–31599(4 席) | 同上 | **hold 登録**(`p10-bench-4p`、閾値なし) |
| 昇格 1 回目 dev 31600–32399(3 席) | 同上 | **hold 登録**(`p10-promote1-3p`、閾値 1/3。D5「CI 下限 > 1/n」) |
| 昇格 1 回目 dev 32400–33199(4 席) | 同上 | **hold 登録**(`p10-promote1-4p`、閾値 0.25) |

設計 D4 は「旧 2 人予約列を無断で消費しない」と定めているため、旧案の dev 帯は登録しなかった。
2026-10-09 の裁定で多人数専用 dev 領域 30000–39999 を新設し(上の宣言行)、4 帯をそこへ置き換えて hold 登録した。
hold は帯・人数・用途・閾値を固定する。対戦(heuristic×(n−1) / 凍結した初期多人数 policy×(n−1) 等)と
モデル SHA は C6 で同じ id のまま確定する。
なお Phase 5 は 20000 番台をこの列の外として使っており、本台帳の「上限なし」の記述と整合していない(未裁定、記録のみ)。

### C6 の消費と結果(2026-10-10、B-C6 t_c508a571)

各帯は 1 回だけ使った。再実行・選り直しはしていない。機械可読の予約/消費行は下の JSON 欄にある。
promote は greedy(人数別 checkpoint)。accept は強(同モデル + ISMCTS H=2、c_puct=1.5、T=4500)。

| 予約 id | 帯 | 対戦(正準順、先頭が対象) | 局 | 勝率 [95% CI] | 判定 |
|---|---|---|---:|---|---|
| `p10-bench-3p` | dev 30000–30799 | n3 initial / gen199 × {heuristic×2, initial×2} の 4 対戦 | 各 2400 | initial: h 0.0000, i 0.3333 / gen199: h 0.8904 [0.8779, 0.9025], i 1.0000 | ベンチ(閾値なし) |
| `p10-bench-4p` | dev 30800–31599 | n4 同上 | 各 3200 | initial: h 0.0000, i 0.2500 / gen199: h 0.9375 [0.9287, 0.9463], i 1.0000 | ベンチ(閾値なし) |
| `p10-promote1-3p` | dev 31600–32399 | n3 gen199, initial×2 | 2400 | 0.9988 [0.9971, 1.0000] | > 1/3 → **promote** |
| `p10-promote1-4p` | dev 32400–33199 | n4 gen199, initial×3 | 3200 | 1.0000 [1.0000, 1.0000] | > 1/4 → **promote** |
| `p10-accept-3p` | accept 800–1599 | strong:n3 gen199, heuristic×2 | 2400 | 0.9633 [0.9558, 0.9704] | > 0.60 → **合格** |
| `p10-accept-4p` | accept 1600–2399 | strong:n4 gen199, heuristic×3 | 3200 | 0.9631 [0.9563, 0.9697] | > 0.50 → **合格** |

成果物: `docs/eval-runs/phase10/promotion/`(n{3,4}-round1.*)、`docs/eval-runs/phase10/accept/`(n{3,4}.json/.jsonl)、
凍結モデル `docs/eval-runs/phase10/models/champion.json`。全 summary は保存 records だけからの再計算で一致を確認済み。
初期 champion は再生成物で、本学習の学習前重みとのバイト同一性は未証明(間接証拠のみ、`initial-champion.json`)。

### C3 の小規模確認(train。判定ではない)

| 予約 id | 人数 | 対戦(正準順) | group / 範囲 | クラスタ / 局 |
|---|---:|---|---|---:|
| `p10-c3-smoke-3p` | 3 | heuristic, random, random | `train` / 870000–870019 | 20 / 60 |
| `p10-c3-smoke-4p` | 4 | heuristic, random, random, random | `train` / 870100–870119 | 20 / 80 |

`train` 870000–870119 は Phase 7 PPO 対照群の連番(〜821455)と新規学習の開始帯(900000 以降)の間。
大型ネットの連番(200000 以降)の現在位置は台帳に無いため重なりの有無は未確認だが、train は判定に使わない。

<!-- MULTI-SEAT-RESERVED:BEGIN -->
{"group":"accept","hi":1599,"id":"p10-accept-3p","lo":800,"matchups":[["strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","heuristic","heuristic"]],"model_sha256":{"strong:docs/eval-runs/phase10/models/n3/gen199.meta.json heuristic heuristic":{"strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"}}},"num_players":3,"power":{"alpha_one_sided_per_judgement":0.024998,"alternative":0.65,"approximate_power":0.8074197244805748,"bonferroni_judgements":2,"ci":"95% two-sided percentile cluster bootstrap; pass iff ci_low > threshold","clusters":800,"delta":0.05,"family_alpha_bound":0.049996,"note":"normal approximation; not a finite-sample bootstrap guarantee","required_clusters":786,"sigma":0.5,"target_power":0.8,"threshold":0.6,"z_alpha":1.96,"z_beta":0.842},"purpose":"phase10-final-accept-3p","registered_at":"2026-10-09T20:09:43Z","threshold":0.6}
{"group":"accept","hi":2399,"id":"p10-accept-4p","lo":1600,"matchups":[["strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","heuristic","heuristic","heuristic"]],"model_sha256":{"strong:docs/eval-runs/phase10/models/n4/gen199.meta.json heuristic heuristic heuristic":{"strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"}}},"num_players":4,"power":{"alpha_one_sided_per_judgement":0.024998,"alternative":0.55,"approximate_power":0.8074197244805748,"bonferroni_judgements":2,"ci":"95% two-sided percentile cluster bootstrap; pass iff ci_low > threshold","clusters":800,"delta":0.05,"family_alpha_bound":0.049996,"note":"normal approximation; not a finite-sample bootstrap guarantee","required_clusters":786,"sigma":0.5,"target_power":0.8,"threshold":0.5,"z_alpha":1.96,"z_beta":0.842},"purpose":"phase10-final-accept-4p","registered_at":"2026-10-09T20:09:43Z","threshold":0.5}
{"group":"train","hi":870019,"id":"p10-c3-smoke-3p","lo":870000,"matchups":[["heuristic","random","random"]],"model_sha256":{"heuristic random random":{}},"num_players":3,"power":{"alpha_one_sided_per_judgement":0.024998,"alternative":0.55,"approximate_power":0.06516696166520974,"bonferroni_judgements":2,"ci":"95% two-sided percentile cluster bootstrap; pass iff ci_low > threshold","clusters":20,"delta":0.05,"family_alpha_bound":0.049996,"note":"normal approximation; not a finite-sample bootstrap guarantee","required_clusters":786,"sigma":0.5,"target_power":0.8,"threshold":0.5,"z_alpha":1.96,"z_beta":0.842},"purpose":"phase10-c3-smoke","registered_at":"2026-10-09T02:12:11Z","threshold":0.5}
{"group":"train","hi":870119,"id":"p10-c3-smoke-4p","lo":870100,"matchups":[["heuristic","random","random","random"]],"model_sha256":{"heuristic random random random":{}},"num_players":4,"power":{"alpha_one_sided_per_judgement":0.024998,"alternative":0.55,"approximate_power":0.06516696166520974,"bonferroni_judgements":2,"ci":"95% two-sided percentile cluster bootstrap; pass iff ci_low > threshold","clusters":20,"delta":0.05,"family_alpha_bound":0.049996,"note":"normal approximation; not a finite-sample bootstrap guarantee","required_clusters":786,"sigma":0.5,"target_power":0.8,"threshold":0.5,"z_alpha":1.96,"z_beta":0.842},"purpose":"phase10-c3-smoke","registered_at":"2026-10-09T02:12:11Z","threshold":0.5}
{"group":"dev","hi":30799,"id":"p10-bench-3p","lo":30000,"matchups":[["checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","heuristic","heuristic"],["checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json"],["checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json","heuristic","heuristic"],["checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json"]],"model_sha256":{"checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"},"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"}},"checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json heuristic heuristic":{"checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"}},"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"}},"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json heuristic heuristic":{"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"}}},"num_players":3,"power":null,"purpose":"phase10-initial-bench-3p","registered_at":"2026-10-09T20:03:58Z","threshold":null}
{"group":"dev","hi":31599,"id":"p10-bench-4p","lo":30800,"matchups":[["checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","heuristic","heuristic","heuristic"],["checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json"],["checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json","heuristic","heuristic","heuristic"],["checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json"]],"model_sha256":{"checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"},"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"}},"checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json heuristic heuristic heuristic":{"checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"}},"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"}},"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json heuristic heuristic heuristic":{"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"}}},"num_players":4,"power":null,"purpose":"phase10-initial-bench-4p","registered_at":"2026-10-09T20:03:58Z","threshold":null}
{"group":"dev","hi":32399,"id":"p10-promote1-3p","lo":31600,"matchups":[["checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json"]],"model_sha256":{"checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"},"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"}}},"num_players":3,"power":{"alpha_one_sided_per_judgement":0.024998,"alternative":0.3833333333333333,"approximate_power":0.8074197244805748,"bonferroni_judgements":2,"ci":"95% two-sided percentile cluster bootstrap; pass iff ci_low > threshold","clusters":800,"delta":0.05,"family_alpha_bound":0.049996,"note":"normal approximation; not a finite-sample bootstrap guarantee","required_clusters":786,"sigma":0.5,"target_power":0.8,"threshold":0.3333333333333333,"z_alpha":1.96,"z_beta":0.842},"purpose":"phase10-promote1-3p","registered_at":"2026-10-09T20:03:58Z","threshold":0.3333333333333333}
{"group":"dev","hi":33199,"id":"p10-promote1-4p","lo":32400,"matchups":[["checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json"]],"model_sha256":{"checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"},"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"}}},"num_players":4,"power":{"alpha_one_sided_per_judgement":0.024998,"alternative":0.3,"approximate_power":0.8074197244805748,"bonferroni_judgements":2,"ci":"95% two-sided percentile cluster bootstrap; pass iff ci_low > threshold","clusters":800,"delta":0.05,"family_alpha_bound":0.049996,"note":"normal approximation; not a finite-sample bootstrap guarantee","required_clusters":786,"sigma":0.5,"target_power":0.8,"threshold":0.25,"z_alpha":1.96,"z_beta":0.842},"purpose":"phase10-promote1-4p","registered_at":"2026-10-09T20:03:58Z","threshold":0.25}
{"frozen":false,"frozen_at":null,"gates":{"G1":{"lineup":"checkpoint:candidate (greedy) + strong:B x (n-1)","pass":"ci97.5_low > 1/3","statistic":"target win rate (ranking[target_seat]==0)","threshold":"1/3"},"G2":{"lineup":"strong:candidate (new net + frozen search) + strong:B x (n-1)","pass":"ci97.5_low > 8/15 (excess q > 3/10)","requires":"nothing; R4 is evaluated only after G2 passes","statistic":"target win rate (ranking[target_seat]==0)","threshold":"8/15"}},"group":"dev","hi":34199,"id":"p12b-dev-g-3p","kind":"G","lo":33200,"matchups":[],"model_sha256":{},"num_players":3,"power":{"G1_threshold":"1/3","G2_excess_rate":"3/10","G2_threshold":"8/15","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-dev-G1G2-3p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":false,"frozen_at":null,"gates":{"G1":{"lineup":"checkpoint:candidate (greedy) + strong:B x (n-1)","pass":"ci97.5_low > 1/4","statistic":"target win rate (ranking[target_seat]==0)","threshold":"1/4"},"G2":{"lineup":"strong:candidate (new net + frozen search) + strong:B x (n-1)","pass":"ci97.5_low > 19/40 (excess q > 3/10)","requires":"nothing; R4 is evaluated only after G2 passes","statistic":"target win rate (ranking[target_seat]==0)","threshold":"19/40"}},"group":"dev","hi":35199,"id":"p12b-dev-g-4p","kind":"G","lo":34200,"matchups":[],"model_sha256":{},"num_players":4,"power":{"G1_threshold":"1/4","G2_excess_rate":"3/10","G2_threshold":"19/40","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-dev-G1G2-4p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":false,"frozen_at":null,"gates":{"R4":{"dev_reject":"point < -5/100","order":"after G2 passed, same alpha 0.0125, intersection-union","pass":"point >= 0 and ci97.5_low > -3/100 for every bench (AND)","statistic":"paired d = p_candidate - p_B per bench (same deal, opponents, seat map)"}},"group":"dev","hi":36199,"id":"p12b-dev-r4-3p","kind":"R4","lo":35200,"matchups":[],"model_sha256":{},"num_players":3,"power":{"G1_threshold":"1/3","G2_excess_rate":"3/10","G2_threshold":"8/15","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-dev-R4-3p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":false,"frozen_at":null,"gates":{"R4":{"dev_reject":"point < -5/100","order":"after G2 passed, same alpha 0.0125, intersection-union","pass":"point >= 0 and ci97.5_low > -3/100 for every bench (AND)","statistic":"paired d = p_candidate - p_B per bench (same deal, opponents, seat map)"}},"group":"dev","hi":37199,"id":"p12b-dev-r4-4p","kind":"R4","lo":36200,"matchups":[],"model_sha256":{},"num_players":4,"power":{"G1_threshold":"1/4","G2_excess_rate":"3/10","G2_threshold":"19/40","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-dev-R4-4p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":false,"frozen_at":null,"gates":{"EXPLOIT":{"hole_line_diagnostic":"ci95_low of exploiter win rate > 1/n + 5/100","note":"undetected means undetected within the search budget","pass":"point <= 0 and ci97.5_high < 3/100 (added to R4)","statistic":"paired d_E = p_E(vs candidate) - p_E(vs B) per frozen exploiter"}},"group":"dev","hi":38199,"id":"p12b-dev-ex-3p","kind":"EXPLOIT","lo":37200,"matchups":[],"model_sha256":{},"num_players":3,"power":{"G1_threshold":"1/3","G2_excess_rate":"3/10","G2_threshold":"8/15","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-dev-exploit-3p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":false,"frozen_at":null,"gates":{"EXPLOIT":{"hole_line_diagnostic":"ci95_low of exploiter win rate > 1/n + 5/100","note":"undetected means undetected within the search budget","pass":"point <= 0 and ci97.5_high < 3/100 (added to R4)","statistic":"paired d_E = p_E(vs candidate) - p_E(vs B) per frozen exploiter"}},"group":"dev","hi":39199,"id":"p12b-dev-ex-4p","kind":"EXPLOIT","lo":38200,"matchups":[],"model_sha256":{},"num_players":4,"power":{"G1_threshold":"1/4","G2_excess_rate":"3/10","G2_threshold":"19/40","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-dev-exploit-4p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":false,"frozen_at":null,"gates":{"EXPLOIT":{"hole_line_diagnostic":"ci95_low of exploiter win rate > 1/n + 5/100","note":"undetected means undetected within the search budget","pass":"point <= 0 and ci97.5_high < 3/100 (added to R4)","statistic":"paired d_E = p_E(vs candidate) - p_E(vs B) per frozen exploiter"},"G1":{"lineup":"checkpoint:candidate (greedy) + strong:B x (n-1)","pass":"ci97.5_low > 1/3","statistic":"target win rate (ranking[target_seat]==0)","threshold":"1/3"},"G2":{"lineup":"strong:candidate (new net + frozen search) + strong:B x (n-1)","pass":"ci97.5_low > 8/15 (excess q > 3/10)","requires":"nothing; R4 is evaluated only after G2 passes","statistic":"target win rate (ranking[target_seat]==0)","threshold":"8/15"},"R4":{"dev_reject":"point < -5/100","order":"after G2 passed, same alpha 0.0125, intersection-union","pass":"point >= 0 and ci97.5_low > -3/100 for every bench (AND)","statistic":"paired d = p_candidate - p_B per bench (same deal, opponents, seat map)"}},"group":"accept","hi":3399,"id":"p12b-accept-3p","kind":"ACCEPT","lo":2400,"matchups":[],"model_sha256":{},"num_players":3,"power":{"G1_threshold":"1/3","G2_excess_rate":"3/10","G2_threshold":"8/15","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-accept-G1G2R4-3p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":false,"frozen_at":null,"gates":{"EXPLOIT":{"hole_line_diagnostic":"ci95_low of exploiter win rate > 1/n + 5/100","note":"undetected means undetected within the search budget","pass":"point <= 0 and ci97.5_high < 3/100 (added to R4)","statistic":"paired d_E = p_E(vs candidate) - p_E(vs B) per frozen exploiter"},"G1":{"lineup":"checkpoint:candidate (greedy) + strong:B x (n-1)","pass":"ci97.5_low > 1/4","statistic":"target win rate (ranking[target_seat]==0)","threshold":"1/4"},"G2":{"lineup":"strong:candidate (new net + frozen search) + strong:B x (n-1)","pass":"ci97.5_low > 19/40 (excess q > 3/10)","requires":"nothing; R4 is evaluated only after G2 passes","statistic":"target win rate (ranking[target_seat]==0)","threshold":"19/40"},"R4":{"dev_reject":"point < -5/100","order":"after G2 passed, same alpha 0.0125, intersection-union","pass":"point >= 0 and ci97.5_low > -3/100 for every bench (AND)","statistic":"paired d = p_candidate - p_B per bench (same deal, opponents, seat map)"}},"group":"accept","hi":4399,"id":"p12b-accept-4p","kind":"ACCEPT","lo":3400,"matchups":[],"model_sha256":{},"num_players":4,"power":{"G1_threshold":"1/4","G2_excess_rate":"3/10","G2_threshold":"19/40","alpha_one_sided":0.0125,"approximate_power":0.8214421362669176,"clusters":1000,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-accept-G1G2R4-4p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.3654095568750873,"clusters":1000,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:14Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":true,"frozen_at":"2026-10-10T19:41:52Z","gates":{"EXPLOIT":{"hole_line_diagnostic":"ci95_low of exploiter win rate > 1/n + 5/100","note":"undetected means undetected within the search budget","pass":"point <= 0 and ci97.5_high < 3/100 (added to R4)","statistic":"paired d_E = p_E(vs candidate) - p_E(vs B) per frozen exploiter"},"G1":{"lineup":"checkpoint:candidate (greedy) + strong:B x (n-1)","pass":"ci97.5_low > 1/3","statistic":"target win rate (ranking[target_seat]==0)","threshold":"1/3"},"G2":{"lineup":"strong:candidate (new net + frozen search) + strong:B x (n-1)","pass":"ci97.5_low > 8/15 (excess q > 3/10)","requires":"nothing; R4 is evaluated only after G2 passes","statistic":"target win rate (ranking[target_seat]==0)","threshold":"8/15"},"R4":{"dev_reject":"point < -5/100","order":"after G2 passed, same alpha 0.0125, intersection-union","pass":"point >= 0 and ci97.5_low > -3/100 for every bench (AND)","statistic":"paired d = p_candidate - p_B per bench (same deal, opponents, seat map)"}},"group":"train","hi":19900001,"id":"p12b-c1-smoke-3p","kind":"SMOKE","lo":19900000,"matchups":[{"challenger":"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json"],"role":"G1"},{"challenger":"strong:docs/eval-runs/phase10/models/n3/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json"],"role":"G2"},{"challenger":"strong:docs/eval-runs/phase10/models/n3/initial.meta.json","opponents":["heuristic","random"],"role":"R4"},{"challenger":"strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","opponents":["heuristic","random"],"role":"R4"},{"challenger":"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n3/initial.meta.json","strong:docs/eval-runs/phase10/models/n3/initial.meta.json"],"role":"EXPLOIT"},{"challenger":"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json"],"role":"EXPLOIT"}],"model_sha256":{"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json,strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"},"strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"}},"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/initial.meta.json,strong:docs/eval-runs/phase10/models/n3/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"},"strong:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"}},"G1|checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json,strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"},"strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"}},"G2|strong:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json,strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"},"strong:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"}},"R4|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json|heuristic,random":{"strong:docs/eval-runs/phase10/models/n3/gen199.meta.json":{"meta":"cfb40fb2c5492f1d1b6cfeb508eb0cc3cec111da18282161b5201c5fc48f4e87","weights":"c20eccb50d535c7a59eec511071e206577250edd4c66a968159901fd6b3dadcd"}},"R4|strong:docs/eval-runs/phase10/models/n3/initial.meta.json|heuristic,random":{"strong:docs/eval-runs/phase10/models/n3/initial.meta.json":{"meta":"fa3df006661b1c5da48c43c4da031a7d95b4621b953c030edbee4c70ee93a9ea","weights":"5e8998d701a0f8eb99c7cbaea40b13102e555e95a739cba3cd42c618e899023c"}}},"num_players":3,"power":{"G1_threshold":"1/3","G2_excess_rate":"3/10","G2_threshold":"8/15","alpha_one_sided":0.0125,"approximate_power":0.017865239933082866,"clusters":2,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-c1-smoke-3p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.015520375333551995,"clusters":2,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:15Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
{"frozen":true,"frozen_at":"2026-10-10T19:47:44Z","gates":{"EXPLOIT":{"hole_line_diagnostic":"ci95_low of exploiter win rate > 1/n + 5/100","note":"undetected means undetected within the search budget","pass":"point <= 0 and ci97.5_high < 3/100 (added to R4)","statistic":"paired d_E = p_E(vs candidate) - p_E(vs B) per frozen exploiter"},"G1":{"lineup":"checkpoint:candidate (greedy) + strong:B x (n-1)","pass":"ci97.5_low > 1/4","statistic":"target win rate (ranking[target_seat]==0)","threshold":"1/4"},"G2":{"lineup":"strong:candidate (new net + frozen search) + strong:B x (n-1)","pass":"ci97.5_low > 19/40 (excess q > 3/10)","requires":"nothing; R4 is evaluated only after G2 passes","statistic":"target win rate (ranking[target_seat]==0)","threshold":"19/40"},"R4":{"dev_reject":"point < -5/100","order":"after G2 passed, same alpha 0.0125, intersection-union","pass":"point >= 0 and ci97.5_low > -3/100 for every bench (AND)","statistic":"paired d = p_candidate - p_B per bench (same deal, opponents, seat map)"}},"group":"train","hi":19900101,"id":"p12b-c1-smoke-4p","kind":"SMOKE","lo":19900100,"matchups":[{"challenger":"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json"],"role":"G1"},{"challenger":"strong:docs/eval-runs/phase10/models/n4/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json"],"role":"G2"},{"challenger":"strong:docs/eval-runs/phase10/models/n4/initial.meta.json","opponents":["heuristic","random","random"],"role":"R4"},{"challenger":"strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","opponents":["heuristic","random","random"],"role":"R4"},{"challenger":"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/initial.meta.json"],"role":"EXPLOIT"},{"challenger":"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","opponents":["strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json"],"role":"EXPLOIT"}],"model_sha256":{"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"},"strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"}},"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/initial.meta.json,strong:docs/eval-runs/phase10/models/n4/initial.meta.json,strong:docs/eval-runs/phase10/models/n4/initial.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"},"strong:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"}},"G1|checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"},"strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"}},"G2|strong:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"},"strong:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"}},"R4|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json|heuristic,random,random":{"strong:docs/eval-runs/phase10/models/n4/gen199.meta.json":{"meta":"6f9509fd0ff4ab6f918e7d7d041f46bc8ff8453d927fc9f4390992e69f911473","weights":"0261685ccecd24996354372959fa4f0d98665bb3571063fc1ea54073f05f161c"}},"R4|strong:docs/eval-runs/phase10/models/n4/initial.meta.json|heuristic,random,random":{"strong:docs/eval-runs/phase10/models/n4/initial.meta.json":{"meta":"5843b30ac4697c2d35cd556f3d75b15880668661cd714fb98d70e76bd9e54c54","weights":"d3c705fceb43394b68ad4f1e709601f9f46ff80841a9ff8bdc266943a2dacaec"}}},"num_players":4,"power":{"G1_threshold":"1/4","G2_excess_rate":"3/10","G2_threshold":"19/40","alpha_one_sided":0.0125,"approximate_power":0.017865239933082866,"clusters":2,"delta":0.05,"family_alpha":0.05,"family_judgements":"G1/G2 x n3/n4 (Bonferroni)","note":"normal approximation for G1/G2 only; not a finite-sample bootstrap guarantee and not the R4 power","power_at_800_not_used":0.7214063392787158,"required_clusters":951,"sigma":0.5,"target_power":0.8,"z_alpha":2.2414027276049464,"z_beta":0.8416212335729144},"protocol":"phase12b-multi/v1","purpose":"phase12b-c1-smoke-4p","r4_sensitivity":{"alpha_one_sided":0.0125,"ci_condition_pass_probability":0.015520375333551995,"clusters":2,"margin":0.03,"note":"probability that only the CI condition (97.5% low > -0.03) holds at true difference 0; excludes the point-estimate condition and the AND over all benches; separate from the G1/G2 power","sigma_paired_cluster_diff":0.5,"true_diff":0.0},"registered_at":"2026-10-10T19:41:15Z","search":{"algorithm":"ismcts","budget_seconds":null,"budget_transitions":4500,"c_puct":1.5,"evaluator":"NetEvaluator(same model)","fpu":0.5,"horizon":2,"pending_delegate":"PolicyAgent(same model, greedy, seed=0)","root_policy":"puct","root_seed":"policy_seed = derive_seed(root,'policy',index,seat)"}}
<!-- MULTI-SEAT-RESERVED:END -->

<!-- MULTI-SEAT-CONSUMED:BEGIN -->
{"group":"train","hi":870019,"id":"p10-c3-smoke-3p","launched_at":"2026-10-09T02:12:12Z","lo":870000,"matchup":["heuristic","random","random"],"run_id":"1a0a4aaf66cc645f55817566f378de68e720d1ea271c9febd90c3e73297f5cb3"}
{"group":"train","hi":870119,"id":"p10-c3-smoke-4p","launched_at":"2026-10-09T02:12:12Z","lo":870100,"matchup":["heuristic","random","random","random"],"run_id":"1e6ce35cd1663b1516a9ed8df16013d657de672e283b68c66e261eb7c19f28bf"}
{"group":"dev","hi":30799,"id":"p10-bench-3p","launched_at":"2026-10-09T20:05:28Z","lo":30000,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","heuristic","heuristic"],"run_id":"77f1e0bd6470da507d980ad45a318499c153a9350472db0bd5868d4a2a145d13"}
{"group":"dev","hi":30799,"id":"p10-bench-3p","launched_at":"2026-10-09T20:05:36Z","lo":30000,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json"],"run_id":"ab1b74b947a4eeb6b91677254aa24cabc6e8fe6c3245efca874ba7ea302dd035"}
{"group":"dev","hi":30799,"id":"p10-bench-3p","launched_at":"2026-10-09T20:05:51Z","lo":30000,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json","heuristic","heuristic"],"run_id":"d0638c8e1adb84e5004b19063d317e2c25d61c1d1d9ba313b76b39ea11eefc9a"}
{"group":"dev","hi":30799,"id":"p10-bench-3p","launched_at":"2026-10-09T20:05:59Z","lo":30000,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json"],"run_id":"9f73d67753b42f40e5bf95d34107c18764b43eb792f079c502f63f36e4fc5803"}
{"group":"dev","hi":32399,"id":"p10-promote1-3p","launched_at":"2026-10-09T20:06:08Z","lo":31600,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json"],"run_id":"09cbb8e7ca2caba962fc31b446b15285941cf650b542fdd0ea09c452cf7346d9"}
{"group":"dev","hi":31599,"id":"p10-bench-4p","launched_at":"2026-10-09T20:06:17Z","lo":30800,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","heuristic","heuristic","heuristic"],"run_id":"1b348be3dd3673670ff6d1aea759204fbb8d2057c0ccdd922d0651ecc7ce4fee"}
{"group":"dev","hi":31599,"id":"p10-bench-4p","launched_at":"2026-10-09T20:06:27Z","lo":30800,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json"],"run_id":"66f43a22e88a3f46e840a578abcbbdbb48dd2848b1d473e6523fab994f306bef"}
{"group":"dev","hi":31599,"id":"p10-bench-4p","launched_at":"2026-10-09T20:06:43Z","lo":30800,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json","heuristic","heuristic","heuristic"],"run_id":"0765ed7cfb5db2eb1aec77d89e7eaf4f277ddf13b5156c1946c3d50198a407af"}
{"group":"dev","hi":31599,"id":"p10-bench-4p","launched_at":"2026-10-09T20:06:52Z","lo":30800,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json"],"run_id":"b8c13af42be6acbf2b0f1b25051d0913156ea56ba8b334a819844a099dad35a2"}
{"group":"dev","hi":33199,"id":"p10-promote1-4p","launched_at":"2026-10-09T20:07:03Z","lo":32400,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/gen199.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json"],"run_id":"090c122353f47f6c5917d59627c5eeb09bf1367ea5a64631392022eb91d2cb00"}
{"group":"accept","hi":1599,"id":"p10-accept-3p","launched_at":"2026-10-09T20:10:08Z","lo":800,"matchup":["strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","heuristic","heuristic"],"run_id":"6876c3b40cfee36259aeb0c192da54f5aaf52359b54f323d2fca56b39368304c"}
{"group":"accept","hi":2399,"id":"p10-accept-4p","launched_at":"2026-10-09T20:10:11Z","lo":1600,"matchup":["strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","heuristic","heuristic","heuristic"],"run_id":"c65200cb20fe65263af867f611d0de90168181da9b5fb404d7e6236bb095abf2"}
{"group":"train","hi":19900001,"id":"p12b-c1-smoke-3p","launched_at":"2026-10-10T19:41:52Z","lo":19900000,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json"],"matchup_key":"G1|checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json,strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","protocol":"phase12b-multi/v1","role":"G1","run_id":"c37cb0780d8bb2748c1cc7de7b411c45b2d02ba0dfaf5e5c86db2e4657971e51"}
{"group":"train","hi":19900001,"id":"p12b-c1-smoke-3p","launched_at":"2026-10-10T19:42:20Z","lo":19900000,"matchup":["strong:docs/eval-runs/phase10/models/n3/initial.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json"],"matchup_key":"G2|strong:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json,strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","protocol":"phase12b-multi/v1","role":"G2","run_id":"c2a820e15aecd27e97aaaf8f327846c81c22f03be188339712e18dc85179ae18"}
{"group":"train","hi":19900001,"id":"p12b-c1-smoke-3p","launched_at":"2026-10-10T19:43:28Z","lo":19900000,"matchup":["strong:docs/eval-runs/phase10/models/n3/initial.meta.json","heuristic","random"],"matchup_key":"R4|strong:docs/eval-runs/phase10/models/n3/initial.meta.json|heuristic,random","protocol":"phase12b-multi/v1","role":"R4","run_id":"47886c895c8dade47386d49e39b74ccfbbd637bc46db1dacdd1e2c414eb8f7c3"}
{"group":"train","hi":19900001,"id":"p12b-c1-smoke-3p","launched_at":"2026-10-10T19:44:04Z","lo":19900000,"matchup":["strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","heuristic","random"],"matchup_key":"R4|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json|heuristic,random","protocol":"phase12b-multi/v1","role":"R4","run_id":"d36a2f612068ea9a6aaf7393b2080544ec91c33e9c42a0426f4429d17f599ffc"}
{"group":"train","hi":19900001,"id":"p12b-c1-smoke-3p","launched_at":"2026-10-10T19:44:23Z","lo":19900000,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","strong:docs/eval-runs/phase10/models/n3/initial.meta.json","strong:docs/eval-runs/phase10/models/n3/initial.meta.json"],"matchup_key":"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/initial.meta.json,strong:docs/eval-runs/phase10/models/n3/initial.meta.json","protocol":"phase12b-multi/v1","role":"EXPLOIT","run_id":"5a07a6d92c028b4d26914107d172fffad2101d229dc01d3e6720b4da44507942"}
{"group":"train","hi":19900001,"id":"p12b-c1-smoke-3p","launched_at":"2026-10-10T19:45:13Z","lo":19900000,"matchup":["checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","strong:docs/eval-runs/phase10/models/n3/gen199.meta.json"],"matchup_key":"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n3/initial.meta.json|strong:docs/eval-runs/phase10/models/n3/gen199.meta.json,strong:docs/eval-runs/phase10/models/n3/gen199.meta.json","protocol":"phase12b-multi/v1","role":"EXPLOIT","run_id":"6bb939ff4a77ba5e372911a981ec6d275c88e7d73b5974b6c2e359ce4e1e2b77"}
{"group":"train","hi":19900101,"id":"p12b-c1-smoke-4p","launched_at":"2026-10-10T19:47:44Z","lo":19900100,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json"],"matchup_key":"G1|checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","protocol":"phase12b-multi/v1","role":"G1","run_id":"811701fbeb3e8f0f0943fc16fe17ced52eea4e86c61a7e630d6aea4aa1d9aeda"}
{"group":"train","hi":19900101,"id":"p12b-c1-smoke-4p","launched_at":"2026-10-10T19:48:57Z","lo":19900100,"matchup":["strong:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json"],"matchup_key":"G2|strong:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","protocol":"phase12b-multi/v1","role":"G2","run_id":"a572d503904912f4445d2fcf10108e703068710d7d742793f0a5a8d506686285"}
{"group":"train","hi":19900101,"id":"p12b-c1-smoke-4p","launched_at":"2026-10-10T19:51:45Z","lo":19900100,"matchup":["strong:docs/eval-runs/phase10/models/n4/initial.meta.json","heuristic","random","random"],"matchup_key":"R4|strong:docs/eval-runs/phase10/models/n4/initial.meta.json|heuristic,random,random","protocol":"phase12b-multi/v1","role":"R4","run_id":"b13390076fa27e75a54afbe2e7ea638afc313b2151abc688b8867709d0e83763"}
{"group":"train","hi":19900101,"id":"p12b-c1-smoke-4p","launched_at":"2026-10-10T19:52:45Z","lo":19900100,"matchup":["strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","heuristic","random","random"],"matchup_key":"R4|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json|heuristic,random,random","protocol":"phase12b-multi/v1","role":"R4","run_id":"6ef954298a79b8d6f3264fb41d3d24830eb144d1692f6efc8dea01f1f9713a93"}
{"group":"train","hi":19900101,"id":"p12b-c1-smoke-4p","launched_at":"2026-10-10T19:53:15Z","lo":19900100,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/initial.meta.json"],"matchup_key":"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/initial.meta.json,strong:docs/eval-runs/phase10/models/n4/initial.meta.json,strong:docs/eval-runs/phase10/models/n4/initial.meta.json","protocol":"phase12b-multi/v1","role":"EXPLOIT","run_id":"36ae37045fa1c8aa67800e2db3e63ea64d0803d7ab66ffac7bbf387bc0c3286f"}
{"group":"train","hi":19900101,"id":"p12b-c1-smoke-4p","launched_at":"2026-10-10T19:55:14Z","lo":19900100,"matchup":["checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","strong:docs/eval-runs/phase10/models/n4/gen199.meta.json"],"matchup_key":"EXPLOIT|checkpoint:docs/eval-runs/phase10/models/n4/initial.meta.json|strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json,strong:docs/eval-runs/phase10/models/n4/gen199.meta.json","protocol":"phase12b-multi/v1","role":"EXPLOIT","run_id":"312335bc3c36b254db13a9f9a07223d239c5c90823f368cceedd4fe0891598ca"}
<!-- MULTI-SEAT-CONSUMED:END -->

### C5 train-only lifecycle/profile and main training (t_f27f2aa5)

- Verification: n=3 train 930000–930191; n=4 train 940000–940191. Repeated worker=1/2/4 runs deliberately reuse the same logical episodes to test exact determinism. SIGTERM interrupted rollout reuses its unconsumed index on resume.
- Promotion/initial-benchmark smoke: train 970300–970303 (3 seats), 970400–970403 (4 seats), four cyclic clusters per matchup. Diagnostic only, no dev/accept consumption or certification. Initial policy is frozen before any PPO update.
- Main runs are sequential, never concurrent: n=3 train 1000000–1204799; n=4 train 1300000–1504799, at most 200 generations × 1024 episodes each. Each stage additionally stops at 14400 seconds; interrupted rollout does not advance next_episode_index. Fixed manifests and source/conditions SHA are in docs/eval-runs/phase10/jobs/.
- All four thread environments are 1; main workers=32 (aggregate ≤96). Models are player-count-specific, pool categories 50% latest / 25% last five saved generations / 25% heuristic; absent past reallocates to latest. No certified model is produced by training or smoke. C6 owns recovery/formal promotion/final accept.


## Phase 12b — `phase12b-multi/v1` の事前登録(2026-10-11、C1 t_1cc2cca7)

判定規則・検出力・帯の用途は `docs/phase12b-protocol.md`(設計 `docs/phase12b-design.md` D1/D4)が正本。結果を見てから変えない。
機械可読行は上の `MULTI-SEAT-RESERVED` / `MULTI-SEAT-CONSUMED` 欄にあり、`"protocol":"phase12b-multi/v1"` 欄で旧 D4 行と区別する。
旧 D4 経路(`eval_agent.py --num-players`)は protocol 付きの行を拒否し、新経路(`eval_agent.py phase12b`)は protocol の無い行を拒否する。
帯の重複検査は同じ関数(`ledger_conflicts`)で 2 人戦列・Phase 10・既存行と共有する。

**用途拡張(多人数専用 dev 領域 30000–39999)**: 2026-10-09 裁定の現行用途は「3・4 席の初期固定ベンチと昇格判定」だけだった。
本登録で同領域の 33200–39199 に Phase 12b の正式 dev 判定(G1/G2、R4 固定回帰ベンチ、凍結搾取者の診断と d_E)を追加する
(設計 D1「多人数 dev の用途拡張を追記してから登録する」)。2 人昇格の予約列・`promote._next_reserved_direct_band` は無変更。
39999 を越える追加は割り当てず設計へ戻す。

| id | 帯 | 種別 | 判定(合否は 97.5% 両側 percentile クラスタ bootstrap、95% も報告) |
|---|---|---|---|
| `p12b-dev-g-3p` | dev 33200–34199 | G | G1: 下限 > 1/3。G2: 下限 > 8/15 |
| `p12b-dev-g-4p` | dev 34200–35199 | G | G1: 下限 > 1/4。G2: 下限 > 19/40 |
| `p12b-dev-r4-3p` | dev 35200–36199 | R4 | 各ベンチ paired d: 点推定 >= 0 かつ下限 > -0.03(-0.05 未満は dev 棄却) |
| `p12b-dev-r4-4p` | dev 36200–37199 | R4 | 同上 |
| `p12b-dev-ex-3p` | dev 37200–38199 | EXPLOIT | d_E: 点推定 <= 0 かつ上限 < +0.03。穴の線(診断)95% 下限 > 1/n+0.05 |
| `p12b-dev-ex-4p` | dev 38200–39199 | EXPLOIT | 同上 |
| `p12b-accept-3p` | accept 2400–3399 | ACCEPT | 正式 G1/G2/R4(+d_E)の事前固定対戦列を 1 回だけ |
| `p12b-accept-4p` | accept 3400–4399 | ACCEPT | 同上 |

- 全 8 帯は 1000 クラスタ(n3 3000 局、n4 4000 局)、再検査で重なりなし(既消費 accept 0–2399、dev 30000–33199 Phase 10)。
- family alpha <= 0.05(G1/G2 × n3/n4)→ 片側 0.0125。G1/G2 検出力(正規近似、Delta=0.05、sigma=0.5): 必要 951、採用 1000、power ≈ 0.8214(800 なら ≈ 0.7214)。
- **R4 の CI 感度は別記**: paired 差 sigma=0.5、真の差 0、1000 クラスタ、alpha=0.0125、境界 -0.03 で CI 条件だけの通過確率 ≈ 0.3654。
  点推定条件や全ベンチ AND を含む全体通過率ではない。dev/accept を見て境界・局数を変えない。各行の `power` / `r4_sensitivity` に保存。
- 全行 `frozen=false`(**hold、起動不可**)。候補・B・ベンチ・搾取者・探索の SHA は L-DEV / L-ACC が `phase12b freeze` で各 1 回だけ固定する。
  正式帯は hold 確定 + 候補/相手/探索 hash 凍結後に 1 回だけ。peek・再調整・閾値緩和・欠損隠しはしない。

### C1 基盤 smoke(train。判定ではない)

| id | 人数 | group / 範囲 | クラスタ |
|---|---:|---|---:|
| `p12b-c1-smoke-3p` | 3 | `train` / 19900000–19900001 | 2(6 対戦 × 6 局) |
| `p12b-c1-smoke-4p` | 4 | `train` / 19900100–19900101 | 2(6 対戦 × 8 局) |
| (台帳外、G3 診断) | 3 / 4 | `train` / 19900230–19900231 / 19900240–19900241 | 各 2 局 |

`train` 19,900,000 台は既存台帳に使用記録がなく、Phase 12b の train ブロック案(2,000,000–13,999,999)・診断案(20,000,000 以降)と重ならない。
対戦は initial を「候補」に置いた配線確認で、結果は性能を意味しない。成果物は `docs/eval-runs/phase12b/c1/`。dev/accept は消費していない。


## Phase 12b — train/診断ブロック事前登録(LEDGER-TRAIN t_e096de07)

登録/campaign_started_utc: `2026-10-11T15:55:43Z`(裁定後に再固定したhandoff起点)。campaign_deadline_utc: `2026-10-23T23:55:43Z`(296h、カード待ちも含む)。
正本: `phase12b-execution.md` §5.2/§5.3、`phase12b-design.md` D3/D4/D5。C3R 承認 `80d8e01` 後の事前登録。
全行 train・予約のみ・本番実消費0。本登録は long launch の許可ではない。各L/Rはmanifest/recoveryで
許可区間と実消費を照合、TRAIN-CLOSEが消費を追記する。dev/acceptと既存行はbyte単位で無変更。

### C3 smoke の台帳取込み(既登録の転記、新規本番帯ではない)

`eval-runs/phase12b/smoke/REGISTRATION.md`(事前登録commit `8d4ed04`)の train **19910000–19929999** を履歴として取込む。
決定論/TERM→resumeの同局再生は同書の明示的共有だけ。n3 resume 19925000–19925959、n4 resume 19927000–19927959。
consumerの19928000/19928010/19928020も同帯内。予備全体を消費済みとは主張しない。
既存C1 smoke/G3とも、本登録の全28ブロックとも非交差。

### 本学習の予約区間(両端を含む、forkのみ先頭へ移す)

| 用途 | n3 train | n4 train | 最大実消費index |
|---|---|---|---:|
| A | 2000000–2999999 | 4000000–4999999 | 819200 |
| B | 3000000–3999999 | 5000000–5999999 | 819200 |
| EXB-initial | 6000000–6999999 | 8000000–8999999 | 204800 |
| EXB-target | 7000000–7999999 | 9000000–9999999 | 204800 |
| EXC-initial | 10000000–10999999 | 12000000–12999999 | 204800 |
| EXC-target | 11000000–11999999 | 13000000–13999999 | 204800 |

A/Bは各800×1024、EXB/EXCは開始ごと200×1024。EXのinitialはPhase10 initial、targetは旧B/凍結候補greedy。
通常resumeは保存next_episode_index継承、A2/B2のために座標をリセットしない。未実施延長・欠損は未使用/未達。

### 診断の予約区間と最大index使用数

indexは配札/クラスタ/局の座標。n巡回・相手順列・同ログ推論は新indexではなく複製単位を別記する。
G1/G4/搾取者比較/ネット選択の下位割当は各200クラスタに固定(正式dev/acceptの1000クラスタ規則は無変更)。
用途の余白は未割当で、追加比較・再調整に流用しない。100000を超える予測/登録外のCLIや卓は投入せず再設計へblock。

| 用途 | n3 train | n4 train | 最大使用index/人数 |
|---|---|---|---:|
| profile | 20000000–20099999 | 21000000–21099999 | 6400 |
| PFSP | 20100000–20199999 | 21100000–21199999 | 2048 |
| G1 | 20200000–20299999 | 21200000–21299999 | 1200 |
| G4 | 20300000–20399999 | 21300000–21399999 | 70400 |
| exploiter | 20400000–20499999 | 21400000–21499999 | 6400 |
| C | 20500000–20599999 | 21500000–21599999 | 9600 |
| G3 | 20600000–20699999 | 21600000–21699999 | 400 |
| kingmaking | 20700000–20799999 | 21700000–21799999 | 100 |

以下のbはn3=20000000、n4=21000000。区間は[start,start+count−1]。
全CLI/table/対象/世代/開始ごとの展開済みindex一覧を `test-results/phase12b/evidence/t_e096de07/index-list.csv` に凍結。
index-list.csv SHA-256: `ff801d1d9c6026841d1cb0755f0abf0b1c06bc96ab1319c062628202fb38ca81`。
行数964(ヘッダ除く)。CSVはこの台帳の下位割当正本の一部で、証拠とともにcommitする。

#### profile(用途ごと10000刻み、全6 clone)

| slot | 対象 | start | 学習局 | consumer |
|---:|---|---|---:|---|
| 0 | A | b | 1024 | train_ppo.py mode A、fork clone |
| 1 | B | b+10000 | 1024 | train_ppo.py mode B、fork clone |
| 2 | EXB-initial | b+20000 | 1024 | train_ppo.py mode exploiter、initial→旧B |
| 3 | EXB-target | b+30000 | 1024 | train_ppo.py mode exploiter、旧B greedy→旧B |
| 4 | EXC-initial | b+40000 | 1024 | train_ppo.py mode exploiter、initial→候補 |
| 5 | EXC-target | b+50000 | 1024 | train_ppo.py mode exploiter、候補 greedy→候補 |

B profileのwindow0 PFSPはb+11024+t×64(t=0..3、各64cl)のみ、profile slot1に内包。
profile時点ではR-EXBの凍結搾取者が未存在のため、卓t=3のexploiter席は計算コストが同じ**代用品**(gen199の凍結コピー、同アーキテクチャ・mode=sample、SHAはprofile manifestに固定)とする(オーケストレータ裁定2026-10-12)。
profileの目的はD5の秒/RSS/lr/入力SHAの追測で本学習の相手選びではない。この代用品のcheckpoint・重み・PFSP重みは本学習Bのpool/開始重みへ持ち込まない(probe checkpoint非置換と同扱い)。本学習B(L-B1)は従来どおりR-EXBの凍結搾取者を読む。DAGは無変更。
本学習PFSP帯へ書かない。PFSP `diag_start_index=b+11024, diag_max_index=b+11279`。
各profile 1世代は本学習の達成世代へ加えず、実秒は各対象32h/12hとcampaignから控除する。

#### PFSP / G1 / G4 / 搾取者比較の下位割当

PFSPの卓順t=0..3をoldB,past,past-mixed,exploiter-oldBに固定。
正準相手は順に[旧B]×(n−1)、[@past]×(n−1)、n3=[@past,heuristic]/n4=[@past,heuristic,random]、
[凍結旧B向けexploiter,旧B×(n−2)]。実consumer(train_ppo.py table_agents)どおり他家の旧B/@past/exploiterはmode=sample、learner席のみ指定mode。@pastの凍結集合はfork入力+到達snapshot(実CLIの座標由来抽選)。
B本学習のpfsp_tablesはこの順、64cl、interval_generations=100、
`diag_start_index=b+100000, diag_max_index=b+102047`。profile用manifestとは別。AはPFSP未使用。
window w=0..7はage=0,100,..700で計測し次100世代だけ使用、start=b+100000+w×256+t×64。
同一世代/卓のresumeは保存windowを読戻し、途中中断は同一入力で未完了の計測だけを再開する。

G1はm=0(A)/1(B)、g=0(開始gen199)/1(age400)/2(age800)、start=b+200000+m×10000+g×200。
greedy learner 1席 vs 旧B strong×(n−1)。開始診断もA/B別帯。
400→800の判断は同じ系統の開始と400だけを比較。SELは最終実施段のG1を読み、再選別用のG1を追加実行しない。

固定ベンチ順t=0..10(全11卓)は次のとおり。Bは常に旧B、initialは現存Phase10 initial。

| t | 卓ID | 正準相手(n−1席) |
|---:|---|---|
| 0 | pure-random | random×(n−1) |
| 1 | pure-heuristic | heuristic×(n−1) |
| 2 | pure-initial | initial greedy×(n−1) |
| 3 | pure-biased:school_first | 学校優先×(n−1) |
| 4 | pure-biased:avoid_hiring | 雇用回避×(n−1) |
| 5 | pure-biased:mine_first | 鉱山優先×(n−1) |
| 6 | mixed-B-random | 旧B greedy, random×(n−2) |
| 7 | mixed-B-heuristic | 旧B greedy, heuristic×(n−2) |
| 8 | mixed-B-biased:school_first | 旧B greedy, 学校優先×(n−2) |
| 9 | mixed-B-biased:avoid_hiring | 旧B greedy, 雇用回避×(n−2) |
| 10 | mixed-B-biased:mine_first | 旧B greedy, 鉱山優先×(n−2) |

G4はm=0(A)/1(B)、g=0..7(age100..800)、t=0..10。
start=b+300000+m×50000+g×5000+t×400。candidate/旧Bの2呼出しだけ同配札200clで明示的paired共有、
initialの呼出しはstart+200から別200cl。`ladder.py --num-players n --group train --start-index start --pairs 200` の
正準nエージェントを上表から構成し、各モデル/卓の勝率・平均順位を保存する(n2 Elo流用なし)。
各CLI内部のtarget巡回は同一indexのreplica。全呼出しのmodel SHAはL時点で凍結する。

搾取者比較はk=0(旧B対象)/1(候補対象)、s=0(initial開始)/1(target greedy開始)、c=0..7。
start=b+400000+k×20000+s×10000+c×200、各200cl。
c順=init,snapshot-age100,snapshot-age200,latest0..latest4。latestは回収時の保存世代番号の昇順、keep_latest=5。
全固定checkpoint(initial+100刻みsnapshot+末尾5)をSHAで重複除去し、存在しない/同SHAのslotは未使用として記録する。
打切り時も存在する全固定checkpointを比較、存在未確認の世代を作らない。slot外の保存を要求するなら投入前block。
learner=搾取者greedy、相手=対象strong×(n−1)、対象/開始/checkpointごと別帯。
`eval_agent.py --num-players n --group train --start-index start --clusters 200` の実在経路を使う。

#### C用途: ネット選択と探索モード比較のCLI/table index

ネット選択の固定ベンチはm=0(A)/1(B)、t=0..10でstart=b+500000+m×10000+t×200。
候補AまたはBのmax^n strongと旧B strongを上の全11卓にpaired共有、A系統とB系統は別帯。
G1差<.01のときだけ固定ベンチ最悪差をtie-breakに使い、同点A。結果欠損はblock、G1へC結果を混ぜない。

探索比較はq=0(max^n)/1(C)、t=0..12、start=b+520000+q×10000+t×200。
全体比較のt=0は単独旧B strong卓、t=1は[旧B strong,heuristic×(n−2)]の固定混合卓。
t=2..12は上の全11固定ベンチ順。各モードの候補/旧Bだけpaired共有し、max^nとCの間はindexを共用しない。
全体+0.02・全固定ベンチ≥−0.03、otherwise max^n(設計D3)は変更しない。
全体はt=0,1の400クラスタ集計、固定ベンチはt=2..12。探索は両モード同じT=4500、候補ネットも同一。
Cの実在consumerはharness/rollout + IsmctsAgent(opponent_mixture=...) API(`smoke/run_smoke.py:smoke_c`)。
Cを指定できるCLIは現時点で無い。架空optionを宣言せず、L-SEL driverのAPI呼出しと各table→CSV行の写像を
manifestで固定しsmall-fixture dry-runする。max^nのCLIはeval_agent/ladder、CはAPIで別出力。
混合ID/重み/model/search SHAはL-SELが実在モデルだけで凍結する(未存在の候補SHAを捏造しない)。

G3は `value_calibration.py --num-players n --group train --start-index b+600000 --games 400`。
凍結候補greedyの同400局ログで旧B/候補を推論(--log-meta/--compare-meta)、追加局なし。
キングメイキングはb+700000–b+700099の100配札のみ。単独B/固定混合の卓別、他家全順列、seeded tie-breakの
replicaで同じ100配札を共有する。順列/seed/席/tableはsidecarへ固定し、replicaを独立index標本数へ加算しない。

### 機械可読予約(既存MULTI-SEAT-*欄とは別、消費欄ではない)

検証器は旧台帳(HEAD 80d8e01)の有限区間と構造化lo/hi行、明示座標を全群で保守的に突合し、C3 smoke原登録も追加突合。
旧台帳末尾の「Phase12bブロック案」は案の自己言及として除外を明示記録する。
日付/root_seed/bootstrap seed/PID/遷移数はindexとして扱わない。「2000以降」等の歴史的開始表記は
停止済みrunの終端を照合し、無限区間への勝手な拡張はしない(§3.3のinventory、台帳の具体的終端≤1504799)。
2人dev予約列の上限なしはtrainへの予約ではなく、正式dev列として無変更。
JSON blocksの各行は[id,num_players,lo,hi]。下位一覧はCSVのSHAと行数で拘束する。

<!-- P12B-TRAIN-RESERVED:BEGIN -->
{"blocks":[["n3-A",3,2000000,2999999],["n3-B",3,3000000,3999999],["n3-EXB-initial",3,6000000,6999999],["n3-EXB-target",3,7000000,7999999],["n3-EXC-initial",3,10000000,10999999],["n3-EXC-target",3,11000000,11999999],["n3-profile",3,20000000,20099999],["n3-PFSP",3,20100000,20199999],["n3-G1",3,20200000,20299999],["n3-G4",3,20300000,20399999],["n3-exploiter",3,20400000,20499999],["n3-C",3,20500000,20599999],["n3-G3",3,20600000,20699999],["n3-kingmaking",3,20700000,20799999],["n4-A",4,4000000,4999999],["n4-B",4,5000000,5999999],["n4-EXB-initial",4,8000000,8999999],["n4-EXB-target",4,9000000,9999999],["n4-EXC-initial",4,12000000,12999999],["n4-EXC-target",4,13000000,13999999],["n4-profile",4,21000000,21099999],["n4-PFSP",4,21100000,21199999],["n4-G1",4,21200000,21299999],["n4-G4",4,21300000,21399999],["n4-exploiter",4,21400000,21499999],["n4-C",4,21500000,21599999],["n4-G3",4,21600000,21699999],["n4-kingmaking",4,21700000,21799999]],"calls_sha256":"ff801d1d9c6026841d1cb0755f0abf0b1c06bc96ab1319c062628202fb38ca81","calls_count":964}
<!-- P12B-TRAIN-RESERVED:END -->

再実行: `~/.venvs/naeco39/bin/python test-results/phase12b/evidence/t_e096de07/verify_ledger.py`。
証跡: 同ディレクトリのbaseline-audit.json / audit.json / index-list.csv。
検証結果: 全28外側区間・全964下位区間、既存履歴/hold/C3帯との交差0、下位区間同士の意図しない重複0、
最大使用G4=70400/100000、衝突/用途超過mutation5件拒否。paired/replica共有はCSVのsharing列に固定。
