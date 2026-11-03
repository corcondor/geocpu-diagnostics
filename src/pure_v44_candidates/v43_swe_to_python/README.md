# v43 SWE の ALU 記録を純 Python の教材候補へ移す検査

元の v43 学習 ZIP は読み取り専用です。学習、GPU 実行、重み変更はありません。

## 原教材の内訳

`Grand_cotraining_v43.zip` の `data/train_v43_knowledge_unified.jsonl` にある SWE 系列は 14,193 行です。元 ZIP の SHA256 は `0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198` です。

| 系列 | 元行数 | 提示された Python 課題と元 trace の構文が一致する行 | 不一致 |
|---|---:|---:|---:|
| `swe_alu_accumulate_sum` | 2,995 | 2,995 | 0 |
| `swe_alu_modulo_loop` | 2,199 | 2,199 | 0 |
| `swe_alu_multiply_power` | 2,148 | 2,148 | 0 |
| `swe_alu_parity_branch` | 1,993 | 1,993 | 0 |
| `swe_alu_conditional_filter_loop` | 1,543 | 209 | 1,334 |
| `swe_alu_bxor_product` | 1,496 | 1,496 | 0 |
| `swe_alu_modulo_expr` | 1,199 | 1,199 | 0 |
| `swe_alu_conditional_filter_py` | 620 | 620 | 0 |
| **計** | **14,193** | **12,859** | **1,334** |

1,334 行の不一致は、提示コードの条件式を独立に読んで計算した最終値と、元教材の回答が違います。例として、条件式が二つなのに元 trace は四つの分岐を実行している行があります。これらを Python 教材へ移してはいけません。表の 12,859 行は構文の候補数であり、元 CPU と純 Python 実行器の**全数照合済み件数ではありません**。構文検査の原票は [`syntax_census_v1.json`](syntax_census_v1.json) です。

[`convert_swe.py`](convert_swe.py) は各候補について元の v43 CPU 回路を一段ずつ実行し、全観測を元 trace と比較します。続いて許可済み命令だけから作った Python コードを純 v44 実行器で一回実行し、すべての観測文字列を照合します。元の教材や提示コードをそのまま Python として実行しません。元 prompt、最終数値、元 ZIP の行番号・行 SHA256、実行器出力の学習除外、真 EOS、2,048 位置上限を保存・検査します。同梱評価問題と一致する prompt は受理しません。

元の SWE target は ALU の trace と最後の数値で、日本語の結論文は元からありません。変換候補は日本語で書かれた問いと元の最終数値を保持します。長い日本語説明を新たに作ったとは数えません。Python ブロックは全段を実行して観測を一括返します。元の命令と観測の交互の形式を同一視せず、全観測の数値・順序を確認しています。

先頭 1,222 元行に含まれる SWE 70 行を試した結果は、元 CPU、純 Python、語彙、EOS、2,048 位置をすべて通った 65 行、提示課題と元回答の不一致で拒否した 5 行です。試験の別出力はローカルの `trial_1222_v2/` にあり、学習には使用していません。全教材の実行照合が完了するまでは、残りの行を受理と報告しません。

## 既存の論理・会話系列について

元教材には `LogiQA2_Official_FOL` 12,047 行、`Yale_FOLIO_FirstOrderLogic` 1,204 行、`JNLI_Logical_Operator` 6 系列計 5,552 行、`formal_logic` 4 系列計 96 行、`Yahoo_JAPAN_JNLI_Full` 19,992 行があります。これらの ALU は形式上の 0/1 を計算しますが、自然文からその 0/1 を選ぶ過程を回路は検証しません。日本語と Python の論理を結ぶには、各行の前提・真偽値・結論を別に照合する必要があります。未照合の論理行は受理していません。

`v43_relational_causal_cot` 15,000 行には ALU trace がなく、日本語の因果説明を Python に強制する必要はありません。`GSM8K_Japanese_Full` 7,329 行の実例には、1 個 10 円の商品を 1 万個買って 30% 割引なのに、答えが 700 円という不一致がありました。この系列も、元の CPU 観測が一致するだけでは受理しません。`JMultiWOZ_Dialogue_State_Tracking` 4,331 行は状態更新の候補ですが、独立した対話スロット照合をまだ終えていません。
