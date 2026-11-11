# ナショナルエコノミー(無印)実装仕様

- rules_version: 0.3.1
- game_edition: base
- supported_players: 2-4
- primary_cpu_target_players: 2
- source_notes: docs/phase0-sources.md
- cards_data: data/cards.json
- cards_schema: data/cards.schema.json
- status: provisional
- last_updated: 2026-07-21

本書は無印（初代）のみを対象とする実装仕様である。出典本文の正本は `docs/phase0-sources.md` であり、本書は採用裁定と状態遷移の正本である。`confirmed` は調査ノートの確定情報、`provisional` は第14章の仮裁定を含む。

> **2026-07-21 の改訂: 0.3.0 → 0.3.1(OPEN-027)** ★
> 第13章に「導出公開(derived-public)」の区分を追加した。**遊び方と状態遷移は変わらない**が、
> `RUL-INFO-001` が許す**情報集合**が変わる(相手が正当に利用できる情報が増える)ため、
> ルール裁定版として `rules_version` を上げる。
> Opus は当初「観測API契約の変更にすぎない」として据え置きを提案したが、
> sol の再レビュー(P25-019)により **「保存済み hash を維持するために版を据え置くのは、
> 非互換変更を検出する版番号の目的と逆である」** と判定され、これに従った。
> 証跡: `docs/reviews/2026-07-21-sol-phase2.5-design-review-2.md`。
>
> **帰結**: `rules_version` は直列化ヘッダに載るため `state_hash` が変わる。
> 0.3.0 で生成した `docs/eval-runs/*.jsonl` の `initial_state_hash` は**歴史的成果物**として
> そのまま保持する(勝敗・配札・seed は不変)。詳細は `docs/eval-seeds.md` の注記。
> 観測側の版は `OBS_VERSION` 0.2.0 → 0.3.0 が別途担う。

## 第1章 適用範囲・用語・表記

#### TERM-001: プレイヤー／開始プレイヤー
- 定義: 時計回りの席順を持つ参加者。開始プレイヤーは各ラウンドの最初の行為者で、同点時の優先権を持つ。
- 公開区分: public。出典: [SRC-001]。関連裁定: [RUL-TURN-002, RUL-SCORE-002]。

#### TERM-002: 職場／公共職場／建物／施設
- 定義: 職場はワーカーを置いて効果を使うカード。公共職場は誰でも利用できる職場。建物は所有者の場にある建物カード、施設は所有者が持つだけで働く建物である。所有中の職場は本人のみ利用できる。
- 公開区分: public。出典: [SRC-001]。関連裁定: [RUL-BUILD-001, RUL-BUILD-002]。

#### TERM-003: サプライ／家計／所持金
- 定義: サプライはゲーム外の金銭置場、家計はゲーム内共有の金銭置場、所持金は各プレイヤーの金銭である。「銀行」は用いない。
- 公開区分: public。両替は各領域の額を変えず額面だけを交換する。出典: [SRC-001]。関連裁定: [RUL-MONEY-001]。

#### TERM-004: 手札／山札／捨て札／消費財
- 定義: 手札は本人だけが内容を知るカード群、山札は順序非公開、建物捨て札は公開、消費財は建物でない手札である。手札内容(建物実体の同一性)と山札順は private、手札枚数・捨て札・**手札の消費財枚数**は derived-public とする（枚数は公開。ユーザー確認済 2026-07-20、消費財枚数は 2026-07-21 の裁定 [OPEN-027]）。出典: [SRC-001]。関連裁定: [RUL-INFO-001, RUL-DECK-001]。未解決事項: [OPEN-004, OPEN-027]。

#### TERM-005: 研修中ワーカー／価値／得点
- 定義: 新規雇用ワーカーは研修中で当ラウンド配置不能、次ラウンド開始時に通常ワーカーになる。価値は建物の印刷資産価値であり、終了時得点とは区別する。出典: [SRC-001]。関連裁定: [RUL-WORKER-003, RUL-SCORE-001]。

数量は非負整数ドル・枚数であり端数はない。カード定義ID（`card_id`）と物理カード実体（`instance_id`、形式はPhase 0.5で定義）を区別する。

## 第2章 コンポーネントと初期状態

| 項目 | 2人 | 3人 | 4人 | 出典 | 状態 |
|---|---:|---:|---:|---|---|
| 初期表向きワーカー/人 | 2 | 2 | 2 | SRC-001 | confirmed |
| 初期所持金（開始プレイヤーから順） | $5,$6 | $5,$6,$7 | $5,$6,$7,$8 | SRC-001,007 | confirmed |
| 初期手札 | 3 | 3 | 3 | SRC-001 | confirmed |
| 採石場・鉱山・学校 | 各1 | 各1 | 各1 | SRC-001 | confirmed |
| 大工の設置枚数 | 1 | 2 | 3 | SRC-001,006,007 | confirmed(ユーザー確認済 2026-07-20) |
| 箱外の大工 | 2 | 1 | 0 | SRC-001,006 | confirmed(ユーザー確認済 2026-07-20) |
| 物理ワーカー/色 | 7 | 7 | 7 | SRC-001 | confirmed |
| 建物山札 | 64 | 64 | 64 | SRC-001 | confirmed |
| 消費財の山 | 24(無限扱い) | 同左 | 同左 | SRC-001 | confirmed |
| 未払い賃金(サプライ) | 15(無限扱い) | 同左 | 同左 | SRC-001 | confirmed |
| 金銭(サプライ、無限扱い) | $1×16, $2×20, $5×12, $10×16 | 同左 | 同左 | SRC-001 | confirmed |

#### RUL-SETUP-001: 初期化
- 処理: 建物64枚をシャッフルして山札、消費財24枚を山、未払い賃金15枚・残ワーカー・金銭をサプライ、家計を$0に置く。初期公共職場を表向きに置き、ラウンドカード9枚をR1が上になる順で裏向きに重ねる。公平な方法で席順と開始プレイヤーを決め、表の資源を配り、各人が山札から3枚引く。
- 状態変更: [TX-001, TX-002, TX-003]。不変条件: [INV-001, INV-002, INV-003]。出典: [SRC-001]。

#### RUL-SETUP-002: 初期公共職場
- 処理: 採石場、鉱山、学校各1枚および人数表の大工を置く。未使用大工は箱外でありゲーム中に戻らない。
- 出典: [SRC-001, SRC-006, SRC-007]。未解決事項: [OPEN-001]。

## 第3章 ラウンド構成

| 順序 | フェーズID | フェーズ | 開始条件 | 終了条件 | 次 |
|---:|---|---|---|---|---|
| 1 | PHASE-ROUND-START | 開始 | R1初期化後、又は前R終了後 | 研修中を表にする | 配置 |
| 2 | PHASE-PLACE | 配置 | 配置可能ワーカーあり | 全員の待機通常ワーカーが0 | 終了処理 |
| 3 | PHASE-END | 終了処理 | 配置終了 | 賃金・カード処理完了 | 次R又は終了 |

#### RUL-ROUND-001: ラウンド開始
- 発火時点: PHASE-ROUND-START。処理: 開始プレイヤーから順に各プレイヤーについて、(1)研修中ワーカーを通常状態に表返す、(2)焼畑上に消費財があれば所有者の手札へ回収し、焼畑を建物捨て札へ移す(EFF-slash-and-burn-02)。その後、開始プレイヤーから配置を開始する。
- 出典: [SRC-001]、ユーザー確認(2026-07-20)。台帳取引: [TX-004, TX-026, TX-021]。

#### RUL-TURN-001: 配置手番
- 前提: アクティブプレイヤーに通常・待機ワーカーがある。処理: 合法な職場を一つ選び、ワーカー1枚を置き、当該効果を解決する。効果の必須コスト又は選択を満たせない場合の配置可否は各EFF及び[OPEN-010]に従う。
- 終了: 時計回りで次の待機ワーカーを持つプレイヤーへ進む。出典: [SRC-001]。台帳取引: [TX-005]。

#### RUL-TURN-002: 手番順と配置終了
- 処理: 開始プレイヤーから時計回りに進め、待機ワーカーを使い切った者を飛ばす。全員が使い切ればPHASE-ENDへ進む。
- 出典: [SRC-001]。

#### RUL-WORKER-001: 占有
- 処理: 職場は各ラウンド1ワーカーまで。ただし `base:mine` は人数無制限、大工はカード実体ごとに1人まで。所有建物は所有者だけが利用できる。売却された建物は公共職場となる。
- 出典: [SRC-001,003]。関連: [EFF-mine-01, RUL-BUILD-002]。

#### RUL-WORKER-002: 雇用上限
- 処理: 各プレイヤーの雇用ワーカー数(通常+研修中)の上限は `min(5 + 所有社宅数, 7)`。上限到達時、雇用効果は空振りする。
- 出典: [SRC-001]。未解決事項: [OPEN-017, OPEN-018]。関連: [EFF-company-housing-01, TX-024]。

#### RUL-WORKER-003: 新規雇用(研修中)
- 処理: 雇用効果はサプライから自色ワーカー1枚を「研修中」(裏)状態で手元に追加する。研修中は当ラウンド配置できないが賃金計算の対象になる。次ラウンドのPHASE-ROUND-STARTで通常状態になる。
- 出典: [SRC-001]。台帳取引: [TX-024, TX-004]。
- 例外: 専門学校(EFF-vocational-school-01)の雇用は研修中ではなく待機状態で追加され、当ラウンドから配置できる(ユーザー確認 2026-07-20)。[TX-027]

## 第4章 ワーカー配置手順

#### RUL-PLACE-001: 合法手
- 条件: (1)待機通常ワーカーがある、(2)職場が利用者に開放される、(3)占有上限未到達、(4)効果の配置前提を満たす。
- 配置前提の境界(仮裁定):
  - **必須コストまたは獲得額を満たせない職場には配置できない**: 手札を捨てるコスト(露店系・工場・自動車工場・レストラン)、建設の実行(大工・建設会社・ゼネコン・開拓民・ニコイチ建設で建設対象またはコスト支払いが不可能な場合)[OPEN-021]、家計からの獲得で家計残額が獲得額未満の場合(珈琲店・レストラン・露店系)[OPEN-003]。
  - **獲得系の空振りは配置を禁止しない**: 山札・捨て札枯渇によるドロー不足、雇用上限到達は配置合法で、可能な部分だけ解決する。[OPEN-010]
- 処理順: 合法性確認→[TX-005]で配置→`on_place` EFF解決→必須の後続選択→次手番。出典: [SRC-001]。未解決事項: [OPEN-010, OPEN-021]。

#### RUL-DECK-001: ドロー
- 処理: 必要枚数ごとに山札上を手札へ移す。山札が空なら次に引く直前に建物捨て札全体をシャッフルして山札にする。両方空なら引ける枚数だけ引き、不足は無視する。設計事務所の「公開」もドローと同じ再構成規則に従う(仮裁定)。[OPEN-011]。
- 台帳取引: [TX-006, TX-007]。出典: [SRC-001]。未解決事項: [OPEN-011]。

## 第5章 公共職場

| workplace_id | 表示名 | 占有規則 | 配置前提 | effect_spec_ids | 出典 |
|---|---|---|---|---|---|
| base:quarry | 採石場 | 1 | なし | EFF-quarry-01 | SRC-003 |
| base:mine | 鉱山 | 無制限 | なし | EFF-mine-01 | SRC-003 |
| base:school | 学校 | 1 | なし(上限時は空振り) | EFF-school-01 | SRC-001,003 |
| base:carpenter | 大工 | 各1 | 通常建設を完了可能[OPEN-021] | EFF-carpenter-01 | SRC-001,003 |
| base:stall | 露店 | 1 | 手札1以上・家計$6以上 | EFF-stall-01 | SRC-003,004 |
| base:market | 市場 | 1 | 手札2以上・家計$12以上 | EFF-market-01 | SRC-003 |
| base:high-school | 高等学校 | 1 | なし | EFF-high-school-01 | SRC-003 |
| base:supermarket | スーパーマーケット | 1 | 手札3以上・家計$18以上 | EFF-supermarket-01 | SRC-003 |
| base:university | 大学 | 1 | なし | EFF-university-01 | SRC-003 |
| base:department-store | 百貨店 | 1 | 手札4以上・家計$24以上 | EFF-department-store-01 | SRC-003 |
| base:vocational-school | 専門学校 | 1 | なし | EFF-vocational-school-01 | SRC-003 |
| base:expo | 万博 | 1 | 手札5以上・家計$30以上 | EFF-expo-01 | SRC-003 |

露店〜万博の物理実体はラウンドカード1〜8の表面である(`cards.json`では `base:round-1`〜`base:round-8` が裏面のラウンド情報と表面の職場を併せ持つ。上表の workplace_id は表面の論理IDとして効果仕様の参照に使う)。ラウンドN終了時にカードNをめくることで表面の職場がR(N+1)から利用可能になる。ラウンドカード9はめくられず、その表面は使用しない。[RUL-RCARD-001, OPEN-005]

## 第6章 建設・カード移動・建物利用

#### RUL-BUILD-001: 通常建設
- 発火: 建設効果。利用者は手札の建物1実体を選び、印刷コスト枚数の手札（建物又は消費財）を選んで捨て、選択建物を自分の場へ移す。消費財自体は建設できない。コストは割引等を適用後の非負整数とする。
- 捨て建物は公開捨て札、捨て消費財は消費財山へ戻す。建設済み職場は所有者のみ、施設は配置不能で恒常効果のみを持つ。
- 台帳: [TX-009, TX-010]。コスト支払いの手札廃棄は TX-009 の1回のみで表す(どの手札を捨てるかの選択は取引ではなく行動パラメータ)。出典: [SRC-001]。

#### RUL-BUILD-002: 売却・公共化
- 発火: 賃金支払い不能時のみ。売却可能な所有建物を選び、建物を公共職場として中央へ移し、サプライから価値と同額を得る。売却不可（`locked`）は選べない。金銭が足りるなら売却してはならない。
- 出典: [SRC-001]。台帳: [TX-011, TX-012]。関連: [RUL-INSOLV-001]。

## 第7章 賃金と家計

#### RUL-WAGE-001: 賃金計算
```text
gross_wage(player) = wage_rate(round) * employed_workers(player)
payable_wage(player) = gross_wage(player)
```
- `employed_workers` は通常・研修中を含む全雇用ワーカー。賃金率R1〜R9は $2,$2,$3,$3,$3,$4,$4,$5,$5。負値・端数はない。
- 出典: [SRC-001]。台帳: [TX-013]。

#### RUL-WAGE-002: 家計への支払い
- 処理: 第9章のプレイヤー順で所持金から家計へ `payable_wage` を払う。全額不能なら[RUL-INSOLV-001]を完全に解決してから次人へ進む。家計の金銭は直ちに職場効果で利用可能。
- 出典: [SRC-001]。

#### RUL-HOUSEHOLD-001: 家計からの獲得
- 処理: 家計から得る効果は指定額の全額を得る。家計残額が指定額未満の場合、その効果を持つ職場には配置できない(配置前提。部分獲得はない)。出典: [SRC-001]、ユーザー裁定(2026-07-20)。未解決事項: [OPEN-003](resolved)。台帳: [TX-014]。

## 第8章 支払い不能処理

#### RUL-INSOLV-001: 賃金不足
- 発火条件: 賃金の強制支払い額が所持金を超える。
- 選択者: 支払うプレイヤー。解決順: (1)売却可能な所有建物を任意順に売却し所持金を得る、(2)売却可能建物がない又は全売却後も不足なら、所持金の全額を家計へ払い、残不足$1ごとに未払い賃金1枚を受け取る。(3)元の賃金処理を終了する。
- 禁止: 賃金不足でない売却、売却不可建物の売却、部分的な未払い賃金。脱落なし。
- 台帳: [TX-011, TX-012, TX-013, TX-015]。不変条件: [INV-001, INV-004]。出典: [SRC-001]。

## 第9章 ラウンド終了・開始プレイヤー・ゲーム終了

#### RUL-ROUND-002: 終了処理
- 処理: 開始プレイヤーから時計回りに各人について、(1)配置ワーカーを待機へ戻す、(2)手札枚数が上限を超えているなら上限枚数まで選んで捨てる、(3)賃金を払う。不足時は[RUL-INSOLV-001]。全員後、R9でなければ最上段ラウンドカードを公開して公共職場へ追加し次ラウンドへ進む。
- 手札上限は `5 + 4 * 所有倉庫数` で、この時点だけ適用する(消費財も手札として数える)。[OPEN-019]。出典: [SRC-001]。台帳: [TX-016, TX-017]。

#### RUL-RCARD-001: ラウンドカード
- 処理: ラウンドカードは裏面にラウンド番号と賃金率、表面(カード1〜8)に公共職場を持つ。R(N)の終了処理で山の一番上のカードN(裏面が現ラウンドを表示していたもの)をめくり、その表面の公共職場をR(N+1)から使用可能にする。R9終了時はめくらずゲームを終了する(カード9の表面は使用しない)。
- 出典: [SRC-001,003,008]。未解決事項: [OPEN-005]。台帳: [TX-018]。

#### RUL-GAMEEND-001: 終了
- 条件: R9の終了処理（ワーカー回収、手札調整、賃金・不足処理）完了。終了後の追加ラウンド、ワーカー表返し、R9表面公開は行わない。
- 出典: [SRC-001]。

## 第10章 得点計算・順位決定

#### RUL-SCORE-001: 得点
```text
final_score(player) = sum(owned_building.value)
                    + end_game_effect_points
                    + player.money
                    - 3 * unforgiven_unpaid_wages
```
- 手札は得点にならない（農協の効果対象を除く）。終了時効果は各EFFを解決後に加算する。出典: [SRC-001,007,009]。台帳: [TX-019]。

#### RUL-SCORE-002: 同点
- 処理: 最高得点同点者に開始プレイヤーがいればその者が勝者。いなければ開始プレイヤーから時計回りに近い者を勝者とする。CPUの終端順位もこの順で決める。
- 出典: [SRC-001]。

## 第11章 カード効果仕様

共通規則: 下表の `on_place` 効果は利用者がcontroller、1配置当たり1回、対象・選択がない場合は `none`、`partial_resolution` は[RUL-DECK-001]又は[RUL-HOUSEHOLD-001]に従う。建設は[RUL-BUILD-001]、雇用は研修中を1枚サプライから追加（上限を超えない）する。[TX-006, TX-007, TX-009, TX-010, TX-014, INV-002〜004]

| card_id | 表示名 | 枚数 | cost | value | 分類/属性 | effect_spec_ids | hook | 出典/状態 |
|---|---|---:|---:|---:|---|---|---|---|
| base:farm | 農場 | 8 | 1 | 6 | workplace,farm | EFF-farm-01 | on_place__gain_consumer_goods | SRC-003,004 confirmed |
| base:slash-and-burn | 焼畑 | 2 | 1 | 0 | workplace,farm,locked | EFF-slash-and-burn-01,EFF-slash-and-burn-02 | on_place__slash_and_burn; on_round_start__slash_and_burn_collect | SRC-003,004 provisional OPEN-022 |
| base:cafe | 珈琲店 | 2 | 1 | 8 | workplace | EFF-cafe-01 | on_place__gain_from_household | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:design-office | 設計事務所 | 4 | 1 | 8 | workplace | EFF-design-office-01 | on_place__design_office | SRC-003,004 confirmed |
| base:factory | 工場 | 8 | 2 | 12 | workplace,factory | EFF-factory-01 | on_place__discard_then_draw | SRC-003,004 confirmed |
| base:orchard | 果樹園 | 3 | 2 | 10 | workplace,farm | EFF-orchard-01 | on_place__orchard | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:construction-company | 建設会社 | 4 | 2 | 10 | workplace | EFF-construction-company-01 | on_place__build | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:warehouse | 倉庫 | 3 | 2 | 10 | facility,locked | EFF-warehouse-01 | passive__hand_limit_bonus | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:company-housing | 社宅 | 2 | 2 | 8 | facility,locked | EFF-company-housing-01 | passive__worker_cap_bonus | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:law-office | 法律事務所 | 1 | 2 | 8 | facility,locked | EFF-law-office-01 | on_game_end__forgive_unpaid_wages | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:plantation | 大農園 | 3 | 3 | 12 | workplace,farm | EFF-plantation-01 | on_place__gain_consumer_goods | SRC-003,004 confirmed |
| base:restaurant | レストラン | 2 | 3 | 16 | workplace | EFF-restaurant-01 | on_place__discard_gain_household | SRC-003,004 confirmed |
| base:pioneer | 開拓民 | 2 | 3 | 14 | workplace | EFF-pioneer-01 | on_place__build_farm_free | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:realtor | 不動産屋 | 2 | 3 | 10 | facility,locked | EFF-realtor-01 | on_game_end__score_per_building | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:agricultural-coop | 農協 | 1 | 3 | 12 | facility,locked | EFF-agricultural-coop-01 | on_game_end__score_per_consumer_good | SRC-003,004 confirmed |
| base:general-contractor | ゼネコン | 3 | 4 | 18 | workplace | EFF-general-contractor-01 | on_place__build_then_draw | SRC-003,004 confirmed |
| base:steel-mill | 製鉄所 | 3 | 4 | 20 | workplace,factory | EFF-steel-mill-01 | on_place__draw_cards | SRC-003,004 confirmed |
| base:chemical-plant | 化学工場 | 2 | 4 | 18 | workplace,factory | EFF-chemical-plant-01 | on_place__chemical_plant | SRC-003,004 confirmed |
| base:mansion | 邸宅 | 1 | 4 | 28 | facility,locked | EFF-mansion-01 | passive__none | SRC-003,004 confirmed |
| base:labor-union | 労働組合 | 1 | 4 | 0 | facility,locked | EFF-labor-union-01 | on_game_end__score_per_worker | SRC-003,004 confirmed |
| base:automobile-factory | 自動車工場 | 3 | 5 | 24 | workplace,factory | EFF-automobile-factory-01 | on_place__discard_then_draw | SRC-003,004 confirmed |
| base:two-for-one | ニコイチ建設 | 2 | 5 | 20 | workplace | EFF-two-for-one-01 | on_place__build_two_same_cost | SRC-003,004,005 confirmed(ユーザー確認済 2026-07-20) |
| base:headquarters | 本社ビル | 1 | 5 | 20 | facility,locked | EFF-headquarters-01 | on_game_end__score_per_locked_building | SRC-003,004 confirmed |
| base:railroad | 鉄道 | 1 | 5 | 18 | facility,locked | EFF-railroad-01 | on_game_end__score_per_factory | SRC-003,004 confirmed |

### 効果登録表（各行は `controller; timing; resolution; ledger; status`）

| EFF-ID | controller; timing | resolution（choices/costs/状態変更） | ledger | 出典/状態 |
|---|---|---|---|---|
| EFF-farm-01 | 利用者; on_place | 消費財山から2枚を手札へ。選択なし、必須。 | TX-020 | SRC-003,004 confirmed |
| EFF-slash-and-burn-01 | 利用者; on_place | 消費財5枚を山からこのカードの上に置く(この時点では獲得しない)。1ゲームにつき当該実体1回。 | TX-025 | SRC-003,004+ユーザー確認 provisional OPEN-022 |
| EFF-slash-and-burn-02 | 所有者; on_round_start | 次ラウンド開始時にカード上の消費財を所有者の手札へ移し、この建物実体を建物捨て札へ移す。R9に配置した場合は次ラウンドがないため本効果は発火せず、**焼畑は所有建物のまま終了し不動産屋・本社ビルの得点対象に数える**[OPEN-022, OPEN-026]。 | TX-026,TX-021 | ユーザー確認(2026-07-20) provisional OPEN-022 |
| EFF-cafe-01 | 利用者; on_place | 配置前提: 家計$5以上。家計から$5を得る。 | TX-014 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-design-office-01 | 利用者; on_place | 最大5枚を順に公開し、利用者が1枚を手札へ、残りを建物捨て札へ。公開できた枚数0なら空振り。 | TX-006,TX-007,TX-022 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-factory-01 | 利用者; on_place | 手札から任意2枚を通常の捨先へ捨て、その後4枚引く。2枚未満なら配置不可。 | TX-009,TX-006 | SRC-003,004 confirmed |
| EFF-orchard-01 | 利用者; on_place | 手札枚数が4未満の間、消費財を得て4枚にする。4枚以上なら0枚。 | TX-020 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-construction-company-01 | 利用者; on_place | 手札建物1枚を選び、印刷コスト-1で[RUL-BUILD-001]を行う。建設できない場合は配置不可[OPEN-021]。 | TX-009,TX-010 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-warehouse-01 | 所有者; passive | 手札上限を所有倉庫ごとに+4。RUL-ROUND-002でのみ適用。 | score_only | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-company-housing-01 | 所有者; passive | 雇用上限を+1。ただし物理枚数7を超えない。 | score_only | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-law-office-01 | 所有者; on_game_end | 未払い賃金最大5枚を得点減算対象から除外。カード実体は移動しない。 | TX-019 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-plantation-01 | 利用者; on_place | 消費財3枚を手札へ。 | TX-020 | SRC-003,004 confirmed |
| EFF-restaurant-01 | 利用者; on_place | 配置前提: 手札1枚以上かつ家計$15以上。手札1枚を通常の捨先へ捨て、家計から$15を得る。 | TX-009,TX-014 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-pioneer-01 | 利用者; on_place | 配置前提: 手札にfarm属性建物がある。1枚をコスト0で建設する。建設できない場合は配置不可[OPEN-021]。 | TX-010 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-realtor-01 | 所有者; on_game_end | 所有場の建物数×3点を加算（自身を含む）。 | TX-019 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-agricultural-coop-01 | 所有者; on_game_end | 手札中の消費財枚数×3点を加算。 | TX-019 | SRC-003,004 confirmed |
| EFF-general-contractor-01 | 利用者; on_place | 通常建設1回を完了後、2枚引く。建設不能なら配置不可(ドロー目的のみの配置は不可)。 | TX-009,TX-010,TX-006 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| EFF-steel-mill-01 | 利用者; on_place | 3枚引く。 | TX-006 | SRC-003,004 confirmed |
| EFF-chemical-plant-01 | 利用者; on_place | 解決開始時の手札が0枚なら4枚、その他は2枚引く。 | TX-006 | SRC-003,004 confirmed |
| EFF-mansion-01 | 所有者; passive | 効果なし。価値28のみ。 | score_only | SRC-003,004 confirmed |
| EFF-labor-union-01 | 所有者; on_game_end | 雇用ワーカー数×6点を加算。 | TX-019 | SRC-003,004 confirmed |
| EFF-automobile-factory-01 | 利用者; on_place | 配置前提: 手札3枚以上。手札3枚を通常の捨先へ捨て、その後7枚引く。 | TX-009,TX-006 | SRC-003,004 confirmed |
| EFF-two-for-one-01 | 利用者; on_place | 配置前提: 同じ印刷コストの建物2枚が手札にあり、かつそのコスト1回分の捨て札(建設対象2枚を除く手札)を支払えること。2枚を選び、コスト1回分だけ捨てて順に建設する。 | TX-009,TX-010 | SRC-003,004,005 confirmed(ユーザー確認済 2026-07-20) |
| EFF-headquarters-01 | 所有者; on_game_end | 所有するlocked建物数×6点を加算。 | TX-019 | SRC-003,004 confirmed |
| EFF-railroad-01 | 所有者; on_game_end | 所有するfactory属性建物数×8点を加算。 | TX-019 | SRC-003,004 confirmed |

### 公共職場・ラウンドカードの効果登録

| card_id / EFF-ID | controller; timing | resolution | ledger | 出典/状態 |
|---|---|---|---|---|
| base:quarry / EFF-quarry-01 | 利用者; on_place | 1枚引き、開始プレイヤーマーカーを利用者へ移す（次ラウンドから開始）。 | TX-006,TX-023 | SRC-003 confirmed(ユーザー確認済 2026-07-20) |
| base:mine / EFF-mine-01 | 利用者; on_place | 1枚引く。占有無制限。 | TX-006 | SRC-003 confirmed(ユーザー確認済 2026-07-20) |
| base:school / EFF-school-01 | 利用者; on_place | 上限未満なら研修中ワーカー1人を雇用、上限なら空振り。 | TX-024 | SRC-001,003 confirmed(ユーザー確認済 2026-07-20) |
| base:carpenter / EFF-carpenter-01 | 利用者; on_place | 通常建設1回。建設できない場合は配置不可[OPEN-021]。 | TX-009,TX-010 | SRC-001,003 confirmed(ユーザー確認済 2026-07-20) |
| base:stall / EFF-stall-01 | 利用者; on_place | 手札1枚を捨て、家計から$6を得る。 | TX-009,TX-014 | SRC-003,004 confirmed(ユーザー確認済 2026-07-20) |
| base:market / EFF-market-01 | 利用者; on_place | 手札2枚を捨て、家計から$12を得る。 | TX-009,TX-014 | SRC-003 confirmed(ユーザー確認済 2026-07-20) |
| base:high-school / EFF-high-school-01 | 利用者; on_place | 雇用ワーカー(研修中含む)が4人未満なら、4人になるまで研修中ワーカーを一括雇用する。4人以上なら空振り[OPEN-023]。 | TX-024 | SRC-003 provisional OPEN-023 |
| base:supermarket / EFF-supermarket-01 | 利用者; on_place | 手札3枚を捨て、家計から$18を得る。 | TX-009,TX-014 | SRC-003 confirmed(ユーザー確認済 2026-07-20) |
| base:university / EFF-university-01 | 利用者; on_place | 雇用ワーカー(研修中含む)が5人未満なら、5人になるまで研修中ワーカーを一括雇用する。5人以上なら空振り[OPEN-023]。 | TX-024 | SRC-003 provisional OPEN-023 |
| base:department-store / EFF-department-store-01 | 利用者; on_place | 手札4枚を捨て、家計から$24を得る。 | TX-009,TX-014 | SRC-003 confirmed(ユーザー確認済 2026-07-20) |
| base:vocational-school / EFF-vocational-school-01 | 利用者; on_place | 上限未満なら労働者1人を待機状態で雇う(研修中ではなく当ラウンドから配置可能。ユーザー確認2026-07-20)。 | TX-027 | SRC-003 confirmed(ユーザー確認済 2026-07-20) |
| base:expo / EFF-expo-01 | 利用者; on_place | 手札5枚を捨て、家計から$30を得る。 | TX-009,TX-014 | SRC-003 confirmed(ユーザー確認済 2026-07-20) |
| base:consumer-goods / EFF-consumer-goods-01 | 所有者; passive | 建設不可。建設コスト又は露店系の捨札に選べ、農協得点の対象。 | TX-009,TX-019 | SRC-001,003 confirmed |
| base:unpaid-wage / EFF-unpaid-wage-01 | 所有者; on_game_end | 免除対象外1枚につき3点減算。 | TX-019 | SRC-001,003 confirmed |

## 第12章 台帳・不変条件

#### TX-001: 初期金銭配布
- trigger: RUL-SETUP-001; category: transfer; resource: money; source_zone: supply; destination_zone: player_money; amount: 人数別表; atomicity: sequential; source_refs: [SRC-001]; status: confirmed。
#### TX-002: 初期ドロー
- trigger: RUL-SETUP-001; category: draw; resource: card; source_zone: deck; destination_zone: hand; amount: 3/人; source_refs: [SRC-001]; status: confirmed。
#### TX-003: 初期配置
- trigger: RUL-SETUP-001; category: place; resource: worker/card; source_zone: supply; destination_zone: player_waiting/public; amount: setup表; source_refs: [SRC-001,006]; status: confirmed。
#### TX-004: 研修解除
- trigger: RUL-ROUND-001; category: transform; resource: worker; source_zone: player_training; destination_zone: player_waiting; amount: all; source_refs: [SRC-001]; status: confirmed。
#### TX-005: ワーカー配置
- trigger: RUL-TURN-001; category: place; resource: worker; source_zone: player_waiting; destination_zone: workplace; amount: 1; source_refs: [SRC-001]; status: confirmed。
#### TX-006: 建物ドロー
- trigger: RUL-DECK-001/EFF; category: draw; resource: card; source_zone: deck; destination_zone: hand; amount: effect; insufficient_resource_behavior: draw_possible_only; source_refs: [SRC-001]; status: confirmed。
#### TX-007: 山札再構成
- trigger: RUL-DECK-001; category: shuffle; resource: card; source_zone: discard; destination_zone: deck; amount: all; source_refs: [SRC-001]; status: confirmed。
#### TX-008: (欠番)
- 2026-07-20のsolレビュー指摘によりTX-009へ統合し廃止。番号は再利用しない(設計書§1.3)。
#### TX-009: 手札廃棄
- trigger: RUL-BUILD-001/EFF/RUL-ROUND-002; category: discard; resource: card; source_zone: hand; destination_zone: discard_or_goods_pile; amount: effect; source_refs: [SRC-001,003]; status: confirmed。
#### TX-010: 建設
- trigger: RUL-BUILD-001/EFF; category: transform; resource: building_state; source_zone: hand; destination_zone: owned_building; amount: 1; source_refs: [SRC-001]; status: confirmed。
#### TX-011: 売却公共化
- trigger: RUL-INSOLV-001; category: transform; resource: building_state; source_zone: owned_building; destination_zone: public_workplace; amount: 1; source_refs: [SRC-001]; status: confirmed。
#### TX-012: 売却代金
- trigger: RUL-INSOLV-001; category: transfer; resource: money; source_zone: supply; destination_zone: player_money; amount: building.value; source_refs: [SRC-001]; status: confirmed。
#### TX-013: 賃金支払い
- trigger: RUL-WAGE-002; category: transfer; resource: money; source_zone: player_money; destination_zone: household; amount: payable; source_refs: [SRC-001]; status: confirmed。
#### TX-014: 家計支給
- trigger: RUL-HOUSEHOLD-001/EFF; category: transfer; resource: money; source_zone: household; destination_zone: player_money; amount: request(家計残額>=requestが配置前提[OPEN-003]); source_refs: [SRC-001]+ユーザー裁定(2026-07-20); status: confirmed。
#### TX-015: 未払い賃金発行
- trigger: RUL-INSOLV-001; category: create; resource: card; source_zone: supply; destination_zone: player_unpaid; amount: shortage; source_refs: [SRC-001]; status: confirmed。
#### TX-016: ワーカー回収
- trigger: RUL-ROUND-002; category: return; resource: worker; source_zone: workplace; destination_zone: player_waiting; amount: all; source_refs: [SRC-001]; status: confirmed。
#### TX-017: 手札上限調整
- trigger: RUL-ROUND-002; category: discard; resource: card; source_zone: hand; destination_zone: discard_or_goods_pile; amount: excess; source_refs: [SRC-001]; status: confirmed。
#### TX-018: ラウンド表面公開
- trigger: RUL-RCARD-001; category: reveal; resource: card; source_zone: round_deck; destination_zone: public_workplace; amount: 1; source_refs: [SRC-001]; status: confirmed。
#### TX-019: 得点算出
- trigger: RUL-SCORE-001/EFF; category: score_only; resource: marker; source_zone: none; destination_zone: score; amount: formula; source_refs: [SRC-001]; status: confirmed。
#### TX-020: 消費財取得
- trigger: EFF-farm-01等; category: transfer; resource: card; source_zone: goods_pile; destination_zone: hand; amount: effect; insufficient_resource_behavior: virtual_issue(山が空でも必要枚数の消費財実体を新規発行する。物理カード不足は代用品で表す); source_refs: [SRC-001]; status: confirmed。
#### TX-021: 焼畑除去
- trigger: EFF-slash-and-burn-02; category: discard; resource: building_state; source_zone: owned_building; destination_zone: building_discard; amount: 1; source_refs: [SRC-003]; status: confirmed。
#### TX-022: 設計事務所公開
- trigger: EFF-design-office-01; category: reveal; resource: card; source_zone: deck; destination_zone: temporary_public; amount: up_to_5; source_refs: [SRC-003]; status: confirmed。
#### TX-023: 開始プレイヤー移動
- trigger: EFF-quarry-01; category: transform; resource: marker; source_zone: current_player; destination_zone: actor; amount: 1; source_refs: [SRC-003]; status: confirmed。
#### TX-024: 雇用(研修中)
- trigger: EFF-school-01/EFF-high-school-01/EFF-university-01; category: transfer; resource: worker; source_zone: supply; destination_zone: player_training; amount: 効果による(高校・大学は目標人数までの差分); source_refs: [SRC-001,003]; status: confirmed。
#### TX-025: 焼畑積載
- trigger: EFF-slash-and-burn-01; category: transfer; resource: card; source_zone: goods_pile; destination_zone: on_card(焼畑上); amount: 5; insufficient_resource_behavior: virtual_issue; source_refs: ユーザー確認(2026-07-20); status: confirmed。
#### TX-026: 焼畑回収
- trigger: EFF-slash-and-burn-02; category: transfer; resource: card; source_zone: on_card(焼畑上); destination_zone: hand; amount: all; source_refs: ユーザー確認(2026-07-20); status: confirmed。
#### TX-027: 即時雇用
- trigger: EFF-vocational-school-01; category: transfer; resource: worker; source_zone: supply; destination_zone: player_waiting; amount: 1; source_refs: ユーザー確認(2026-07-20); status: confirmed。

#### INV-001: 金銭所在
- 通常時、実物金銭は `supply + household + 全player_money` のいずれかにのみ存在する。額面交換は各領域合計を変えない。サプライ・未払い賃金は無限代用を許す。例外: TX-015はカード発行であり金銭を生成しない。出典: [SRC-001]。
#### INV-002: 建物実体
- 建物64実体は、山札・手札・建物捨て札・所有建物・公共職場・箱外のいずれか一つにのみある。TX-021の移動先は仮裁定。出典: [SRC-001,003]。
#### INV-003: ワーカー所在
- 各色7実体は、サプライ・待機・研修中・職場のいずれか一つにのみある。プレイヤー雇用数は `min(5+社宅数, 7)` 以下。出典: [SRC-001]。未解決事項: [OPEN-017,018]。
#### INV-004: 家計非負
- 家計・所持金は負にならない。家計支給はTX-014の上限まで、賃金不足はTX-015で表す。出典: [SRC-001,006]。未解決事項: [OPEN-003]。
#### INV-005: 占有
- 通常職場カード実体には各ラウンド高々1ワーカー。鉱山のみ無制限。出典: [SRC-001,003]。
#### INV-006: 手札上限時点
- 手札上限はRUL-ROUND-002の調整後のみ満たすべきで、手番中は上限超過を許す。出典: [SRC-001]。未解決事項: [OPEN-019]。
#### INV-007: 消費財は論理的無限資源
- 消費財実体は「消費財の山」「いずれかの手札」「焼畑カード上」のいずれかにのみ存在する。焼畑上の消費財は手札ではなく、手札上限・農協得点の対象外[OPEN-022]。山は無限資源として扱い(物理24枚は保存則の対象外)、取得は[TX-020]で必要枚数を常に満たし、捨てられた消費財は山へ合流して個体性を失う。**消費財は個体性を持たないため`instance_id`を与えず、手札上・焼畑上ともに枚数で保持する**(ユーザー確認 2026-07-20、[OPEN-024])。未払い賃金も同様に枚数で保持する([TX-015]はcreate、サプライは無限扱い)。出典: [SRC-001](FAQ「無くなったら代用」)。

## 第13章 情報公開区分

#### RUL-INFO-001: 公開区分
- 処理: 下表の区分に従う。手札内容と山札の順序・内容のみ非公開。設計事務所等で一時公開されたカードは公開情報として全員が知る(手札に入った後は非公開に戻るが、既知情報として行動履歴に残る)。
- 出典: [SRC-001]、ユーザー確認(2026-07-20)。未解決事項: [OPEN-004](resolved)、[OPEN-027](resolved)。

##### 「非公開」と「導出公開」の区別 ★(OPEN-027、ユーザー裁定 2026-07-21)

本章の `hidden` は**卓上で直接見えないこと**を意味し、「公開情報から論理的に導けないこと」は
意味しない。両者を混同すると、完全記憶のプレイヤーが当然に持つ知識を実装が捨てることになる。

**手札の消費財枚数は公開情報の関数である。** 消費財が手札に入る経路は農場・大農園
(`EFF-farm-01` 等の定数)、果樹園(公開の手札合計から決まる)、焼畑の回収(公開の積載枚数)
の3つに限られ、出ていく経路は捨て札(枚数が公開)だけであり、初期値は 0 である。
したがって全員が卓上の情報だけから各家の消費財枚数を一意に決定できる。

これを **derived-public(導出公開)** として区分し、実装(`Observation`)は各家の消費財枚数を
公開する。**引き続き非公開なのは「手札にある建物がどの実体か」**である。

**表は2軸で読む** ★(OPEN-027)

- **卓上可視性**: そのプレイヤーが物理的に目で見えるか
- **情報集合 / API**: 完全記憶のプレイヤーが公開情報から**正当に決定できる**か。
  実装(`Observation`)が公開してよいのはこの列が `public` / `derived-public` のものである

**実装者が参照するのは「情報集合 / API」の列である。** 卓上可視性が `hidden` でも
情報集合が `derived-public` なら、`Observation` に載せてよい(むしろ載せるべきである)。

| 状態要素 | 卓上可視性(本人 / 他家) | 情報集合 / API | 備考 |
|---|---|---|---|
| 手札の**建物実体の同一性** | visible / **hidden** | **hidden** | 真に非公開。RUL-INFO-001 |
| 手札枚数(合計) | visible / visible | public | OPEN-004 |
| 手札の消費財枚数 | visible / **hidden** | **derived-public** | 増減経路がすべて公開量 [OPEN-027] |
| 手札の建物枚数 | visible / **hidden** | **derived-public** | 合計 − 消費財枚数 [OPEN-027] |
| 所持金 | visible / visible | public | OPEN-004 |
| 家計・サプライ額 | visible / visible | public | SRC-001 |
| 建物・公共職場・配置ワーカー | visible / visible | public | SRC-001 |
| 建物捨て札 | visible / visible | public | SRC-001 |
| **建物山札**の内容・順序 | hidden / hidden | **hidden** | 真に非公開。SRC-001 |
| 建物山札の枚数 | visible / visible | public | 数えられる |
| 山札再構成の回数 | visible / visible | **derived-public** | 全員が見た事象の回数 [OPEN-027] |
| 再構成以降の各人のドロー枚数 | visible / visible | **derived-public** | 全員が見た事象の回数 [OPEN-027] |
| 再構成で山札に入った実体の集合 | visible / visible | public | 捨て札を全員が見てから混ぜる。RUL-DECK-001 |
| **ラウンドカード列**の内容・順序 | hidden / hidden | **derived-public** | 初期配列が固定でシャッフルしない [OPEN-027] |
| 箱外の大工(`out_of_box`) | visible / visible | **derived-public** | 人数から一意。RUL-SETUP-002 |
| 開始プレイヤーマーカーの予約先 | visible / visible | public | 卓上のマーカー。TX-023 |
| 未払い賃金枚数 | visible / visible | public | RUL-INFO-001 |

> 「山札」は**建物山札**(順序が真に非公開)と**ラウンドカード列**(順序が固定で公開)を
> 明確に区別する。前者を隠し、後者を隠さない。

## 第14章 未解決事項

2026-07-20 ユーザー確認(docs/phase0-open-answers.md)により OPEN-001〜021 は全件 resolved
(resolved_by=ユーザー, resolved_at=2026-07-20。OPEN-002のみ公式脚注により確定)。
反映で派生した新規論点2件(OPEN-022, OPEN-023)を provisional で追加(owner=fable,
user_confirmation_required=true)。
Phase 0.5 のAPI設計とsolレビューで派生した3件(OPEN-024〜026)を 2026-07-20 のユーザー確認で
resolved として追加(owner=Opus)。
Phase 2.5 の設計とsolレビュー(P25-004)で派生した1件(OPEN-027)を 2026-07-21 のユーザー裁定で
resolved として追加(owner=Opus)。
状態別集計: discovered 0 / researched 0 / provisional 2 / awaiting_user 0 / resolved 25 / rejected 0 / superseded 0。

| ID | 論点 | 裁定(実装状態遷移) | 状態 |
|---|---|---|---|
| OPEN-001 | 大工枚数 | 2人=1枚/3人=2枚/4人=3枚。残りは箱外。 | resolved |
| OPEN-002 | 終了順 | RUL-ROUND-002を開始プレイヤーから1人ずつ完全解決する。 | resolved(公式) |
| OPEN-003 | 家計不足 | 家計残額が獲得額未満の職場には配置できない。部分獲得はない。 | resolved(仮裁定を修正) |
| OPEN-004 | 公開区分 | 所持金・手札枚数・未払い賃金枚数は公開、手札内容のみ非公開。 | resolved |
| OPEN-005 | R9表面 | 職場は印刷されておらず、めくられない。 | resolved |
| OPEN-006 | 焼畑 | 配置時に消費財5枚をカード上へ置き、次ラウンド開始時に手札へ回収して焼畑は建物捨て札へ。 | resolved(仮裁定を修正) |
| OPEN-007 | 果樹園 | 手札4枚以上なら空振り。配置は合法。 | resolved |
| OPEN-008 | 法律事務所 | 終了時に未払い賃金最大5枚を-3点の対象から除外。 | resolved |
| OPEN-009 | 不動産屋 | 終了時の所有建物(自身含む・売却済み除く)×3点。 | resolved |
| OPEN-010 | 空振り配置 | 獲得系は対象がなくても配置合法、可能な部分だけ解決。 | resolved |
| OPEN-011 | 完全枯渇ドロー | 捨て札再構成後も空なら引けた枚数だけで終了。 | resolved |
| OPEN-012 | 採石場 | カード1枚を引き、開始プレイヤーマーカーを得る(次ラウンドから)。 | resolved |
| OPEN-013 | 鉱山 | カード1枚を引く。同一ラウンドに何人でも配置可。 | resolved |
| OPEN-014 | 高校・大学 | 高校=労働者を4人にする(2〜3人時に差分を一括雇用)、大学=5人にする(2〜4人時に差分を一括雇用)。到達済みなら空振り。 | resolved(仮裁定を修正) |
| OPEN-015 | 万博名称 | 「万博」。 | resolved |
| OPEN-016 | 表記揺れ | 「ニコイチ建設」「珈琲店」。 | resolved |
| OPEN-017 | 社宅上限 | 雇用上限は5+社宅数(物理上限7)。 | resolved |
| OPEN-018 | 学校系上限 | 雇用上限時は空振り(配置合法)。 | resolved |
| OPEN-019 | 手札上限時点 | ラウンド終了処理のみで適用。消費財も手札に数える。 | resolved |
| OPEN-020 | 割引建設 | 建設会社=max(0,cost-1)、開拓民=0、ニコイチ=同cost2枚にcost1回分。 | resolved |
| OPEN-021 | 建設系空振り | 建設を実行できない建設系職場には配置不可。 | resolved |
| OPEN-022 | 焼畑の細部 | R9に配置した場合は回収機会がなく消費財を獲得しない(終了時に山へ戻る)。積載中の消費財は手札上限・農協得点の対象外。 | provisional(実対局検品で確認) |
| OPEN-023 | 高校・大学の新規労働者の状態 | 研修中で追加(学校と同様、次ラウンドから稼働)。専門学校のみ即時稼働(ユーザー明言)。 | provisional(次回ユーザー確認) |
| OPEN-024 | 消費財・未払い賃金の表現 | 個体性を持たないため`instance_id`を与えず枚数で保持する。INV-007を改訂。 | resolved(ユーザー確認 2026-07-20) |
| OPEN-025 | 手札の建物/消費財の内訳の公開区分 | 内訳は「手札内容」の一部であり**非公開**。公開されるのは手札の合計枚数のみ。 | resolved(第13章に明記) |
| OPEN-026 | R9に配置した焼畑カード本体 | **所有建物のまま終了する**。EFF-slash-and-burn-02は「次ラウンド開始時」の効果でありR9では発火しないため、TX-021も発火しない。したがって不動産屋(EFF-realtor-01)の建物数と本社ビル(EFF-headquarters-01)のlocked建物数に**数える**。 | resolved(ユーザー確認 2026-07-20) |
| OPEN-027 | 手札の消費財枚数は「非公開」か「導出公開」か | 消費財の増減経路はすべて公開量で決まり初期値0であるため、**枚数は公開情報の関数**である。**derived-public** として区分し `Observation` で公開する。非公開のままなのは建物実体の同一性。OPEN-025(物理的に裏面から区別できない)は卓上可視性の裁定として維持し、第13章の表を2軸に分けて併記する。情報集合が変わるため `rules_version` を **0.3.1** へ改訂(sol P25-019)。 | resolved(ユーザー裁定 2026-07-21) |

### CONFLICT正式登録

| ID | 食い違い | 裁定 | 状態 |
|---|---|---|---|
| CONFLICT-001 | 二胡市建設/ニコイチ建設 | 表示名「ニコイチ建設」で確定。 | resolved(ユーザー確認) |
| CONFLICT-002 | コーヒー店/珈琲店 | 表示名「珈琲店」で確定。 | resolved(ユーザー確認) |
| CONFLICT-003 | 農園・農具等のアイコン名 | 内部属性 `farm`(農場・焼畑・果樹園・大農園)。 | resolved(ユーザー確認) |

## 第15章 出典対応表

| 仕様ID | 採用出典 | 補助出典 | 食い違い | 状態 |
|---|---|---|---|---|
| RUL-SETUP-001〜002 | SRC-001 | SRC-006,007 | なし | confirmed/provisional |
| RUL-TURN-001〜002, RUL-WORKER-001〜003 | SRC-001 | SRC-003 | なし | confirmed |
| RUL-BUILD-001〜002, RUL-INSOLV-001 | SRC-001 | SRC-003,004 | なし | confirmed |
| RUL-DECK-001, RUL-HOUSEHOLD-001 | SRC-001 | SRC-006 | なし | provisional |
| RUL-WAGE-001〜002, RUL-ROUND-002 | SRC-001 | SRC-007 | なし | confirmed/provisional |
| RUL-RCARD-001 | SRC-001 | SRC-003,008 | なし | provisional |
| RUL-SCORE-001〜002 | SRC-001 | SRC-007,009 | なし | confirmed |
| EFF-farm-01〜EFF-railroad-01 | SRC-003,004 | SRC-005 | CONFLICT-001〜003 | confirmed/provisional |
| EFF-quarry-01〜EFF-expo-01 | SRC-003 | SRC-001,004 | なし | confirmed/provisional |
| EFF-consumer-goods-01, EFF-unpaid-wage-01 | SRC-001 | SRC-003 | なし | confirmed |
