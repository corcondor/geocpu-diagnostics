# GeoCPU v49 データ監査: family `logic_trace`

監査日: 2026-11-09 / 監査者: 独立データ監査エージェント (家族自身の solver・baselines・test 出力は証拠として使っていない)

対象: `.../wt_v48/research/v49_data_20261108/families/logic_trace/data/{logic_trace_train,logic_trace_dev,logic_trace_heldout}.jsonl`

**判定: ok_with_caveats** (blocker なし。理由は末尾 "Blockers 判定" を参照)

要点 (3 行):
- 84,000 行 (105 サブファミリ x 800) の自作・規則生成データ。実測できた範囲で答えは **全部正しい** (92 サブファミリ、67,080 行を自前 solver で全件検算、不一致 0)。eval 8 種への 24 文字 shingle 漏洩 0、完全文一致 0。
- ただし「数十万」ではない。train 84,000 行 = full_text 17,097,951 字、教師信号 (target+1) 8,857,360 字。しかも名前・物・色を置換して同一視すると 28 ファミリが 800 行未満の実質問題数で、11 ファミリは 100 種未満 (trans_generations は 4 種)。
- T5 はうっすら、T6/T7/T8 は訓練できない (検索なし・知識なし・コード生成なし)。形式も eval と違う (下の 9 章)。

---

## 0. 実行環境・使った道具

全て `python -I -X utf8`、1 回に 1 プロセス、JSONL は 1 行ずつストリーム読み。メモリは数十 MB (ハッシュ集合のみ)。書き込みは `scratchpad/audit_v49/` のみ。既存ファイルの変更・削除なし。

| スクリプト (audit_v49/) | 内容 | 出力 |
|---|---|---|
| `lt_a1_structural.py` | 1,2,3,4 章 (件数・sha256・スキーマ・OOV・重複・テンプレ比率・答え分布) | `lt_a1_out.json`, `lt_a1_view.txt` |
| `lt_a2_solvers.py` + `lt_a2_run.py` | 6 章 自前 solver 92 本 (プロンプト本文のみから計算) | `lt_a2_out.json`, `lt_a2_view.txt` |
| `lt_a3_misc.py` | trace 内の等式 (a op b = c) 全検算 + 表面ショートカット基準 | `lt_a3_out.json`, `lt_a3_shortcut_view.txt`, `lt_a3_dist_view.txt` |
| `lt_a4_leak.py` | 5 章 eval 9 ソースとの 24 文字 shingle / 完全文一致、train-heldout-dev | `lt_a4_out.json`, `lt_a4_view.txt` |
| `lt_a5_heldleak.py`, `lt_a6_heldtrain.py` | shingle から定型句を除いた train-heldout 漏洩の再検査 | `lt_a5_out.json`, `lt_a6_out.json` |
| `lt_a7_diversity.py`, `lt_a8_devstruct.py` | 名前置換を同一視した実質問題数、dev と train の構造重複 | `lt_a7_view.txt`, `lt_a8_out.json` |
| `lt_a9_final.py`, `lt_a10_vocab.py` | 未検算ファミリ一覧・総文字数・bundle vocab との照合 | `lt_a9_out.json`, `lt_a10_out.json` |

vocab_chars.json は存在した (3,498 文字、出典 train_40500_exact.jsonl の full_text 和集合、rows_scanned=423,348)。

---

## 1. Inventory

| split | 行数 | バイト | sha256 | manifest と一致 |
|---|---|---|---|---|
| train | 84,000 | 109,421,949 | 1cb6ef438183c93762607f069b8622d4407cdec335c7eb2d45ac8165b442744b | 行数・バイト・sha256 すべて一致 |
| dev | 1,680 | 2,183,670 | 3b87b7fb4c78d43339707717bff7be6f907d1a135cea423141e9c99e0902484d | 一致 |
| heldout | 1,800 | 2,482,158 | 5c362760dae354a7eaaec6a1d00829022bcba8e19bd47b1ac8fe783ef971662c | 一致 |

- 改行は LF のみ (CR を含む行 0)、不正 JSON 0。
- train は 105 ファミリ x 800 行ちょうど、dev は 105 x 16、heldout は 12 ファミリ (`h_*`) x 150 (manifest の記述どおり)。
- difficulty は 1:2:3:4 = 12.5%:12.5%:37.5%:37.5% (train 10,500/10,500/31,500/31,500)。
- manifest の `vocab_size 3893` は bundle の `base_metadata.json` の 1 文字エントリ数 (3,893) と一致。タスク文の「3,898」は特殊トークン 5 個 (`<bos> <eos> <pad> <sep> <unk>`) を含む総数なので矛盾ではない。

## 2. Schema

必須 10 フィールド (id, prompt, target, full_text, split, family, source, license, tool_steps, loss_mask_spans) の欠落: 全 split で 0。追加フィールド: training_category, structural_template, difficulty, _v46_group, reference_final_answer, trace_lines, trace_final, gen_attempt。

| 検査 | train | dev | heldout |
|---|---|---|---|
| full_text != prompt + target | 0 | 0 | 0 |
| split フィールド != ファイル | 0 | 0 | 0 |
| id 重複 (split 内) | 0 | 0 | 0 |
| id 重複 (split 間) | 0 (全体で 0) | | |
| prompt/target が空 | 0 | 0 | 0 |
| prompt が「問題: 」で始まり「条件と根拠を順に考えてください。」で終わらない | 0 | 0 | 0 |
| target が `\n答え: ...。` で終わらない | 0 | 0 | **1** |
| reference_final_answer != target から抽出した答え | 0 | 0 | 0 |
| 最終 trace 行に答えが含まれない | 0 | 0 | 0 |
| len(full_text)+1 > 2048 | **0** | 0 | 0 |
| tool_steps 非空 / loss_mask_spans 非空 | 0 / 0 (全行空) | 同 | 同 |

heldout の 1 件: `v49logic_heldout_h_string_ops_101`。"czrp" に「最後の1文字を取り除く」等を 5 回行い、最後は空文字列。target 末尾が `答え: 。` (答えが空)。算術的には正しいが、学習例としては退化 (空の答え)。要修正 (その 1 行を除外または再生成)。

長さ (full_text 文字数 = トークン数):

| split | max | mean | p99 | min | 教師信号 sum(len(target)+1) |
|---|---|---|---|---|---|
| train | 1,234 | 203.547 | 781 | 70 | **8,857,360** |
| dev | 1,191 | 204.257 | 777 | 72 | 178,189 |
| heldout | 385 | 221.907 | 360 | 113 | 205,969 |

train の full_text 総文字数 17,097,951。最長ファミリは bool_equiv_two (平均 712、最大 1,234)。context 2048 に対して全行余裕あり (超過 0)。loss_mask_spans が全行空なので、何を教師信号にするかは trainer 既定に依存する (上の数字は依頼どおり target+1 で計算)。

## 3. Vocabulary

- vocab_chars.json (3,498 字) に対する OOV: **0 行 / 0 文字** (train, dev, heldout すべて)。OOV 文字なし。
- 追加検査: bundle `base_metadata.json` の vocab (1 文字エントリ 3,893) に対しても OOV 0。census が報告した「旧コーパスにあって bundle vocab にない 5 文字 (U+5D6F, U+5F3C, U+6350, U+7252, U+7B19)」は本 family では使われていない。
- 使用文字種は train 582 種、全 split で 619 種のみ (モデルの 3,893 種のうち約 16%)。

## 4. Duplicates

| 検査 | 結果 |
|---|---|
| train 内 full_text 完全重複 | 0 (dev 0, heldout 0) |
| 同一 prompt で target が異なる (矛盾答え) | 0 (全 split) |
| 同一 prompt の重複 (答えが同じでも) | 0 |
| train ∩ dev の prompt | 0 |
| train ∩ heldout の prompt | 0 |
| dev ∩ heldout の prompt | 0 |
| train/dev/heldout 間の full_text 一致 | 0 |
| 1 テンプレート (structural_template group) が 15% 超 | なし。最大 `logic_trace/number` 10.5% (8,800 行)。ファミリ単位は各 0.952% (800/84,000) |

**実質的な多様性 (名前置換を同一視)**: カタカナ 2 字以上の連続 (人名等) と物・色の語 (りんご、赤 等) を 1 記号に置換して数えた「実質問題数」(`lt_a7_view.txt`)。train 全体では 84,000 行中 72,597 種 (86.4%)。ただしファミリ別では偏りが大きい。

| ファミリ | 800 行中の実質問題数 |
|---|---|
| trans_generations | 4 |
| round_robin_wins | 6 |
| comb_adjacent_arrangement | 8 |
| trans_graph_reach_count | 12 |
| set_all_chain | 14 |
| liar_2_people | 28 |
| liar_3_accuse | 39 |
| liar_4_culprit | 48 |
| work_together_days | 62 (数の組が 62 通りだけ) |
| syl_region_elim | 64 |
| set_some_all | 67 |
| circle_shift | 147 |
| syl_three_valued | 183 |

100 種未満: 11 ファミリ、200 種未満: 13、400 種未満: 14、800 種未満: 28 ファミリ。つまりこれらのファミリでは「800 行」の大半は同じ問題の名前違いコピー。dev についても、dev 1,680 行のうち 264 行 (15.7%) は名前置換で train と同一の問題で、12 ファミリは dev 16 行すべてが train の問題と名前違い (`lt_a8_out.json`)。dev 精度は一部ファミリでは「汎化」ではなく「名前不変性」しか測らない。

(家族自身の `capacity_probe.txt` は「16 ファミリが 2,400 prompt に達する前に枯渇 (num_from_binary は 1,064 行)」と主張している。これは claim として確認していないが、上の実質問題数の少なさと整合する。数十万行にスケールするにはテンプレート自体の追加が要る。)

## 5. Leakage

### 5a. eval manifest との照合

方法: eval 側は各文字列フィールド (prompt, target, question, choices, final_answer 等。id/pillar/category/subject は除外) から 24 文字 shingle を stride 8 で採取 (`hash()`、165,179 個)。監査対象行は prompt+target (= full_text) を stride 1 で全走査。T8 は `tier8_readonly_v44fix.py` のファイル全文を採取。完全文一致は eval の各行・各文 (。で分割、10 文字以上) と監査行の同様の単位を照合。

eval ソース (件数): benchmark_8100 (8,100; JMMLU+JamC = T3/T4/T7)、swe_step_control (200; T1)、v33_state_tracking (188; T2)、v38_decomposed_state (200)、v41_logic_ops (400; T5)、v42_search_aligned (200; T6)、v42_search_heldout (200; T6)、jamcqa_test (2,309)、tier8 code (4 prompt; T8)。

| 項目 | train | dev | heldout |
|---|---|---|---|
| 上記 9 ソースのいずれかと 24 文字 shingle が 1 つでも一致した行 | **0** | **0** | **0** |
| eval の文・行 (10 字以上) と完全一致した行 | **0** | **0** | **0** |

定型 (generic) shingle (>500 行) と内容一致の切り分けは、eval ソースに対しては一致そのものが 0 なので不要 (examples なし)。陽性対照: eval 本文から 40 文字を埋め込んだ合成行で検出器が 1〜2 hit を返すことを確認 (benchmark_8100 / v41 / v42 の 3 ファイル、`lt_a11_posctrl.json`)。

制限: eval 側 stride 8 のため、24〜30 文字だけの短い共有部分は取りこぼしうる (31 文字以上の共有は必ず検出される)。このデータは語彙が算数・論理の自作文で、JMMLU/JamC/JNLI の文面とは文体が異なる。結論: **eval 項目の train への漏洩は検出されなかった**。

### 5b. train と heldout / dev の漏洩

- heldout 12 ファミリ (`h_*`) は train のどのファミリとも別。prompt の数字・名前・小文字語を置換したテンプレート鍵の一致: heldout 593 種のうち train と一致 **0**。dev は 776 種のうち 472 種が train と一致 (同じファミリなので期待どおり)。
- heldout の行のうち train と完全一致する行 (8 字以上) は、「条件と根拠を順に考えてください。」(1,800 行すべて) と短い `答え: N。` のみ。
- shingle (24 字、定型句を除外): train 84,000 行のうち 191 行 (0.23%) だけが heldout と一致し、一致内容は `0+0=0, 0+1=1, 1+...` のような走査和 trace の断片 (train 側は comb_pairs_sum_target 176 行、num_digit_sum_div 14 行、comb_digits_even_count 1 行。heldout 側はいずれも h_inversion_count の `合計:` 行) で、問題文の共有ではない。heldout 行と dev 行の間も同種の 5 行/2 行のみ。
- 共有されるのは名前プール (ミツル、ユキ 等) と同じ著者の trace 文体で、項目そのものの共有はない。

結論: heldout は train と prompt・テンプレート・問題文を共有していない。ただし heldout は同じ著者・同じ「条件と根拠…」+ trace 形式の別タスクなので、「未知形式への汎化」ではなく「未知タスク種への汎化」を測る。

## 6. 独立した答えの検算

方法: プロンプト本文から正規表現で入力を取り出し、`lt_a2_solvers.py` の自前 solver で答えを再計算して target の「答え: X。」と文字列一致で比較 (家族の solver は import していない)。サンプリングではなく **対象ファミリの全行**を検算。パース失敗・曖昧解 (複数解) は不一致とは別に計上。

結果: **92 ファミリ、67,080 行 (train 80 ファミリ x 800 = 64,000、dev 80 x 16 = 1,280、heldout 12 x 150 = 1,800 から計算; 合計 67,080) で一致 67,080、不一致 0、パース失敗 0、曖昧 0、例外 0**。

依頼されたファミリの内訳 (train 800 / dev 16 / heldout 150 すべて一致):
- sched_*: sched_sequential_end, sched_two_workers, sched_critical_path, sched_meeting_slot, sched_round_robin, sched_next_departure (6 本とも 816/816)
- liar_*: liar_2_people, liar_3_accuse, liar_4_culprit, liar_count_statement (4 本とも 816/816; 全問で解は一意 = 曖昧 0)
- comb_grid_paths 816/816 (DP で再計算)、clock_hand_angle 816/816 (以下/以上の 2 型)
- syl_region_elim 816/816、syl_three_valued 816/816、set_all_chain/set_some_all (三段論法を 8 領域 Venn の反例探索で独立に実装)
- heldout h_* 12 本すべて 150/150: h_assign_match (全順列列挙で唯一解を確認)、h_caesar_shift、h_equation_pair、h_fence_post (4 型)、h_grid_robot、h_inversion_count、h_leap_year_count、h_magic_square、h_mixture_conc、h_pigeonhole、h_string_ops、h_which_true
- そのほか検算済み: arith_* (add_carry, sub_borrow, mul_partial, div_remainder, fraction_add, fraction_compare, mult_chain, precedence)、avg_missing_value、bool_* 7 本、cal_* 3 本、clock_add_duration/elapsed/hand_angle、comb_* 全 8 本、mod_* 全 3 本、num_* 9 本 (num_reverse_diff と num_round_place は未検算)、paren_* 3 本、queue_enq_deq / stack_push_pop_top / stack_rpn_eval、rank_scores_sort、round_robin_wins、seq_* 7 本 (alternating_op, arith_next, fib_like, geo_next, linear_recurrence, nth_term, sum_first_n)、set_* 4 本 (all_chain, some_all, inclusion_count, ops_nested)、shop_change、speed_* 2 本、state_ledger/light_toggle/swap_boxes、work_together_days (内訳合計: train/dev の 80 ファミリ)

**未検算 (自前 solver なし) の 25 ファミリ**: age_offset_chain, circle_shift, clock_gain_loss, line_positions, num_reverse_diff, num_round_place, pct_discount_chain, pct_nested_share, pct_profit_after_discount, pct_reverse_original, rank_chain_extreme, rank_perm_count, ratio_recipe_scale, ratio_split_total, seq_missing_term, seq_second_diff, seq_terms_below, set_attr_filter, trans_cmp_3valued, trans_generations, trans_graph_reach_count, trans_same_group_count, unit_custom_chain, unit_si_chain, unit_speed_convert。これらは "not run" (答えの独立検算なし)。ただし次の trace 内等式検査は全ファミリに適用済み。

**trace 内等式の全検算**: 全 117 ファミリの target から `a op b = c` 連鎖 (数字と + - * / のみ、余り表記は除外) を抽出して Fraction で再計算: 161,936 連鎖、**誤り 0**。(初版の正規表現は `トモ-1=2` のように変数名の直後から連鎖を拾う誤検出を 5,620 件出したが、先頭条件を直して再実行して 0 になった。誤検出は line_positions, age_offset_chain, circle_shift, h_fence_post のみで、実データ側の誤りではない。)

表現上の小さな注意 (答えの誤りではない): num_divisor_count の「約数は全部で何個ありますか。そのうち偶数のものは何個ですか、偶数の個数だけ答えてください。」は文がぎこちない。comb_digits_even_count は名前に反して「偶数/奇数/k の倍数」の 3 型を含む。

## 7. Shortcut / bias audit

(家族の baselines_report.json は主張として参照したのみ。以下は自分の測定: `lt_a3_shortcut_view.txt`, `lt_a3_dist_view.txt`)

答えの分布 (train、ラベル型ファミリ): bool_eval_vars / bool_equiv_two / bool_circuit_eval / bool_xor_fold / liar_2_people / syl_region_elim / set_all_chain / set_some_all / num_prime_test / num_digit_sum_div / paren_balanced は **ちょうど 50:50**。3 値型 (bool_tautology_class, syl_three_valued, trans_cmp_3valued, liar_3_accuse) は **ちょうど 1/3 ずつ** (0.333-0.334)。heldout h_which_true は A 24.7% / B 24.7% / C 25.3% / D 25.3%、正解オプションの値の順位 (小さい順 0-3 位) は 36/37/39/38 件で偏りなし。heldout h_assign_match は ねこ 21.3% … かめ 11.3% (最頻 0.213、一様 0.167)。

heldout の表面ベースライン (12 ファミリ): 最頻回答 (ファミリ内最大) は h_which_true 0.253、h_assign_match 0.213 など最大 0.253。最初/最後/最大/最小の数値や位置規則はほぼ 0 〜 0.13。heldout は偶然水準 (家族の claim と整合)。

train で最頻回答が高いもの (これは答えの事前分布による):

| ファミリ | 最頻回答の比率 | 偶然水準 |
|---|---|---|
| trans_generations | 0.734 (答え 2) | 0.25 |
| rank_perm_count | 0.641 (答え 2) | 0.167 |
| paren_max_depth | 0.466 | 0.143 |
| bool_implies_count | 0.381 | 0.167 |
| comb_pairs_sum_target | 0.381 | 0.111 |
| liar_count_statement | 0.359 | 0.25 |
| trans_same_group_count | 0.359 | 0.20 |

プロンプトから問題を解かずに答えが出せる度合い (交差適合: 偶数 index で閾値を学習し奇数で評価、逆も):

| ファミリ | 規則 | 精度 | 偶然水準 |
|---|---|---|---|
| liar_3_accuse | prompt 長の 1 閾値 | 0.455 | 0.333 |
| syl_three_valued | prompt 長 | 0.380 | 0.333 |
| syl_region_elim | prompt 長 | 0.566 | 0.50 |
| bool_equiv_two | prompt 長 | 0.551 | 0.50 |
| liar_2_people | prompt 長 | 0.532 | 0.50 |
| liar_count_statement | 「答えは prompt 内の k 番目の数」規則 | 0.446 | 0.25 |
| queue_enq_deq | 同上 (答え = 最初の enq の数 等) | 0.309 | 0.111 |
| state_light_toggle | prompt 長 | 0.321 | 0.125 |

(家族自身の同種の数字は、liar_3_accuse 0.52、syl_three_valued 0.479、liar_2_people 0.583 などと、より甘い in-sample 値で報告されており、私の交差適合値より高い。)

判断: ラベル型ファミリでラベルの偏りは除去されていて、heldout も偶然水準。残る弱い長さ手がかり (最大で偶然水準 +12 〜 +13 ポイント: liar_3_accuse) は完全には除去されていないが、manifest が「除去した」と主張しているのは「ラベルの均衡と難度からの独立」であり、これは確認できた。trans_generations (答え 2 が 73%) と rank_perm_count (答え 2 が 64%) は答えの事前分布が偏っている。blocker ではなく caveat。

## 8. Provenance / licence

- source: `selfauthored:geocpu/v49-logic-trace` が train/dev/heldout の 100% (87,480/87,480)、license: `CC0-1.0` が 100%。training_category は全て `logic`。
- 実在の人間データ・外部コーパス由来の行: **0%**。全行が規則生成 (行ごとに (family, index, split, attempt) から決定的に再生成可能と生成器に記載)。
- 出題の型 (つるかめ、年齢算、時計の角、旅人算、仕事算、三段論法、うそつき問題 等) は一般的な算数・論理パズルの定型で、数式は自作。外部問題の転用や MathNet / 教師データとの近似重複の検査は今回 "not run"。CC0 は自己申告であり第三者検証はしていない。

## 9. Format alignment (T5-T8)

この family の行の形式 (全行共通):

```
問題: <問題文>
条件と根拠を順に考えてください。
<根拠の行 x 1〜27>
答え: X。
```
(prompt = `問題: ...\n条件と根拠を順に考えてください。`、target = `\n` + trace 行 + `\n答え: X。`; 平均 trace 4.5 行)

| 評価 | この family で学べるか | eval 形式との差 |
|---|---|---|
| **T5** (6 演算子の論理対比・3 択・JNLI 由来) | 限定的。syl_three_valued (正しい/誤り/どちらとも言えない)、trans_cmp_3valued (はい/いいえ/わからない)、h_which_true (A〜D 選択) が構造的に近い | 評価は自然文: `タスク: 自然言語指示の論理演算子変換・3値論理推論 (JNLI Grounded)\n【本文】（1）テーブルの上に花が活けられています。…\n【指示】次のA〜Cのうち、前提から論理的に【正しい（確実に導ける）】ものを選べ。\n【選択肢】\nA. （3）に関して…\n思考の連鎖と回答:` → 答えは文字 1 つ (`B`)。こちらは架空の「〇〇族」や記号式で、日本語の意味推論 (含意・矛盾・中立) は含まない。プロンプト末尾の合図が異なり (`条件と根拠を順に考えてください。` 対 `思考の連鎖と回答:`)、応答も trace + `答え: B。` で文字 1 つではない |
| **T6** (検索つき多段検索) | **不可**。tool_steps は全行空、検索・検索クエリ・知識事実なし | 評価: `質問: 通信機器大手ノキア（Nokia）が本社を置く国の首都はどこか？\n思考:` (heldout 版は `思考: 外部検索エンジンを使用して多段階で調査します。`) → 答え `ヘルシンキ`。検索クエリ生成の形式が無い |
| **T7** (JMMLU 風の閉/開本 4 択) | **不可**。知識問題ではない (算数・論理のみ)。h_which_true の A〜D は同じ左辺の等式選びで内容が違う | 評価: `分野: abstract_algebra (math)\n問題: ...\nA: 8\nB: 2\nC: 24\nD: 120\n正解:` → `C`。(JamC は `質問: ...\nA: ...\n答え:`)。選択肢が `A: ` 形式、合図が `正解:`/`答え:` |
| **T8** (固定 4 プロンプト: Python/C++/Rust コード + 解説) | **不可**。コードブロック・解説・言語指定が一切ない | 評価: `ユーザー: 関数 f(x) = 3x^3 + 2sin(x) - 5 について、導関数 f'(x) を求め、…\nアシスタント: ` (aligned 版は `問題: ...\n思考:` / `指示: ...\n回答:`)。形式が全く異なる |

まとめ: logic_trace は T6/T7/T8 の学習データにはならず、T5 にも間接的 (論理構造の練習) にしか効かない。応答が「trace + `答え: X。`」なので、T5/T7 の「文字 1 つ」を答えさせる採点とは形式が合わない。他のファミリと混ぜるときは、評価形式に合わせた変換 (または評価側の抽出) が要る。

---

## Blockers 判定

blocker に当たる項目を 1 つずつ確認した結果:

| 項目 | 結果 |
|---|---|
| 間違った答え | 検算した 92 ファミリ・67,080 行で 0 件。trace 内等式 161,936 連鎖で 0 件。未検算は 25 ファミリ (not run) |
| eval 項目の train への漏洩 | 0 (24 字 shingle、完全文一致とも) |
| 矛盾する重複 | 0 |
| 学習できない長すぎる行 | 0 (最大 1,234 字) |
| 壊れたファイル | なし (sha256・行数・バイトすべて manifest 一致、JSON 不正 0) |
| 除去済みと主張しているのに残るショートカット | ラベル均衡・難度独立は確認。弱い長さ手がかりと答えの事前分布の偏りは残る (caveat) |

blocker なし。ただし下記 caveat があるので `ok_with_caveats`。

## Caveats (修正すれば ok)

1. heldout の 1 行 (`v49logic_heldout_h_string_ops_101`) は答えが空文字列 (`答え: 。`)。除外または再生成。
2. 実質問題数が少ない 28 ファミリ (特に 11 ファミリが 100 種未満、trans_generations は 4 種) は、800 行が名前違いの同一問題。dev 1,680 行の 15.7% も train と名前違いで同一。dev 精度の解釈に注意、増やすならテンプレートの追加が必要。
3. trans_generations (最頻 73%)、rank_perm_count (64%) など、答えの事前分布が偏る。長さ手がかりでラベルが +12 〜 +13 ポイント当たるファミリ (liar_3_accuse, syl_region_elim, bool_equiv_two) がある。
4. 25 ファミリは独立検算していない。
5. T6/T7/T8 向けデータではなく、T5 にも形式が合わない (9 章)。「T5〜T8 のためにデータを増やした」には該当しない。
6. 規模は train 84,000 行 / 教師信号 8.86M 字で、数十万ではない。loss_mask_spans が全行空なので、教師信号の範囲は trainer 側の設定で決まる。
7. 検査の限界: 漏洩検査は eval 側 stride 8 の shingle (24〜30 字だけの共有は取りこぼしうる)、provenance の外部照合 (MathNet 等) は未実施。
