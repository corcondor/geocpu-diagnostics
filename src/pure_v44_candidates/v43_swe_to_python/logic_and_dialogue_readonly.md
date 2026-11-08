# v43 の論理・会話教材を Python TIR へ接続できる範囲

元 ZIP `Grand_cotraining_v43.zip` の `data/train_v43_knowledge_unified.jsonl` を読み取り専用で系列別に数えました。これは教材の棚卸しであり、以下の論理・会話系列を Python 教材として**受理した件数は 0 行**です。

| 元系列 | 行数 | 元 ALU trace | 判定 |
|---|---:|---:|---|
| `LogiQA2_Official_FOL` | 12,047 | 12,047 | `SET/IMP/NOT` は教師が選んだ 0/1 を計算する。英語本文からその 0/1 を導く検査は別に必要 |
| `Yale_FOLIO_FirstOrderLogic` | 1,204 | 1,204 | `SET/IMP/XOR` は同様。述語・量化の解釈は回路観測で検証できない |
| `JNLI_Logical_Operator_positive_entailment` | 926 | 926 | 本文と選択肢の含意判断が先に必要 |
| `JNLI_Logical_Operator_not_necessarily_true` | 926 | 926 | 未確定と偽を区別する意味検査が先に必要 |
| `JNLI_Logical_Operator_non_entailment_undeducible` | 925 | 925 | `BXOR` の値だけでは非含意を証明できない |
| `JNLI_Logical_Operator_empty_witness_unsatisfiable` | 925 | 925 | 空の証人の判定は本文の条件確認が先に必要 |
| `JNLI_Logical_Operator_negation_contradiction` | 925 | 925 | `NOT` は本文が含意する値の入力後にしか働かない |
| `JNLI_Logical_Operator_set_complement_not_applicable` | 925 | 925 | 同上 |
| `formal_logic_impl_and` | 72 | 72 | prompt に真偽値が明記されている。厳密な Python 化を試せる第一候補だが未実行 |
| `formal_logic_demorgan` | 8 | 8 | 全真偽割当てで等価性を検査してから判断する |
| `formal_logic_syllogism` | 8 | 8 | 命題変数の推論を個別に検査してから判断する |
| `formal_logic_contrapositive` | 8 | 8 | 同上 |
| `Yahoo_JAPAN_JNLI_Full` | 19,992 | 19,992 | 元 trace は `SET` 一発で答えの符号を置くだけ。Python 実行が言語推論を示すことにはならない |
| `v43_relational_causal_cot` | 15,000 | 0 | 日本語の因果説明をコードなしで保持できる。無理に Python 呼び出しを加えない |

本体回路の値を Python で再現できても、自然文から形式前提へ写す部分は証明されません。そこで論理行を受理するには、元の問・前提の真偽・命題式・結論を独立に照合し、元 CPU と Python の観測も一致させます。LogiQA・FOLIO・JNLI の元 target が正しいかを ALU trace だけで承認しません。元の日本語説明や Lean コードは、検査が済むまで書き換えません。

会話・算数の別系列は `JMultiWOZ_Dialogue_State_Tracking` 4,331 行、`GSM8K_Japanese_Full` 7,329 行です。JMultiWOZ の実例は `SET` による対話スロットの更新ですが、独立した状態照合は未実施です。**v43 の 4,331 行を全走査すると、初期エリア・予算・種類と更新後エリアがすべて一文字になっていました（4,331/4,331 行）。** 実例の日本語は「東周辺で、予算は格な民を探しています」です。元 ZIP の `data/train_v43_knowledge_unified.jsonl` の1始まり47行目、改行を含む原行 SHA256 は `86d0fe5ea7eec06cf886f3a3c180a8b2361731433a6a9179200e85ff478ba701` です。元生成器 [`build_v37_generalized_state_dataset.py`](../../../../../reports/jmmlu_v1/ag_fused_fullqa_training_v1/Grand_cotraining_frontier_v37/build_v37_generalized_state_dataset.py) の49〜61行は京都・格安・民宿などの完全な語を持ちますが、195〜206行で `c1[0]`・`price[0]`・`cat[0]` を使って先頭文字だけを書いています。この生成器の SHA256 は `072216f14e4fb0f526f2f546a1b33e2951b0e65a77d613c579b4c9ac7665d8ab` です。したがって4,331行を自然な日本語会話とPythonの良質な橋として一括採用しません。状態追跡能力の候補としては別に評価する必要があります。

GSM の**一つの**実例は 10 円の商品 1 万個を 30% 割引で買う問題に 700 円と付いており、正しくは 70,000 円です。元 ZIP の同じ学習 member の1始まり13行目、原行 SHA256 は `8eb226f7ded3ea96ac91a175defb708e6f119667729b116e4be1fab09b10f18d` です。元の `SET R0 700` が実行器で再現されても誤教材を移してはいけません。7,329行全体が誤りとは判定していません。

検索系列の `search_augmented_1hop_fact` 4,000 行と `search_augmented_2hop_entity_chain` 3,000 行は、同梱の aligned 評価 200 prompt と完全一致するため全除外です。`v43_search_augmented_general_knowledge` 10,000 行と `v43_academic_multichoice_search_cot` 12,000 行の検索結果は外部ソース依存で、固定 Python 数学実行だけから元観測を再現したとは言えません。この時点では受理していません。

元系列の全件数と同梱評価との照合は [`v43_train_conversion_counts.json`](../../../reports/v44fix/v43_train_conversion_counts.json) と元 ZIP の直接走査で確認しました。元 ZIP の SHA256 は `0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198` です。上記の行数は実行済みの変換件数ではありません。
