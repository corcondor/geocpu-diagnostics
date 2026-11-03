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

元SWE 14,193行の命令列を走査すると、負数の即値と `DIV` は各0行、`MOD` は5,391行に現れました。このSWE候補に負数の除算・剰余があるとは主張しません。受理行は実CPUとPythonの全観測一致で判定します。

[`convert_swe.py`](convert_swe.py) は各候補について元の v43 CPU 回路を一段ずつ実行し、全観測を元 trace と比較します。続いて許可済み命令だけから作った Python コードを純 v44 実行器で一回実行し、すべての観測文字列を照合します。元の教材や提示コードをそのまま Python として実行しません。元 prompt、最終数値、元 ZIP の行番号・行 SHA256、実行器出力の学習除外、真 EOS、2,048 位置上限を保存・検査します。同梱評価問題と一致する prompt は受理しません。

元の SWE target は ALU の trace と最後の数値で、日本語の結論文は元からありません。変換候補は日本語で書かれた問いと元の最終数値を保持します。長い日本語説明を新たに作ったとは数えません。Python ブロックは全段を実行して観測を一括返します。元の命令と観測の交互の形式を同一視せず、全観測の数値・順序を確認しています。

**旧試行の無効化:** `trial_1222_v2/` の65行と `full_verified_v1/` の途中4,325行は、元 CPU と純 Python の数値照合を通りましたが、生成コードのフェンスが行頭ではなく空白の直後にありました。純 v44 の実生成器は `^```python` だけを実行するため、この2出力は教材として無効です。`INVALID_DO_NOT_USE.json` を両方に置いて隔離しました。旧途中ファイルの SHA256 は `77cb938db4893a45f75e73c4176312475236ef3753c07b92407a75102c66bbf4` です。元 ZIP・重み・学習ジョブは変更していません。

新しい変換コードは元の回答境界の先頭空白を保持したうえで改行を足し、コードフェンスを生成文の行頭に置きます。受理前に純 v44 の実際の `PYTHON_BLOCK` で1件のコードブロックとして認識されることを検査します。新しい隔離試行 `trial_1222_native_v3/` は先頭1,222原行中のSWE70行を調べ、元 CPU・純 Python の実実行と独立した保存照合で65行受理、元の課題と命令列の不一致5行を棄却しました。新試行の教材 SHA256 は `6703ce0835de4dab9ab7669a0fcd1972a0e94b71e96b6e7371eea47afe4563f5` です。これは全数の成績ではありません。

## 既存の論理・会話系列について

元教材には `LogiQA2_Official_FOL` 12,047 行、`Yale_FOLIO_FirstOrderLogic` 1,204 行、`JNLI_Logical_Operator` 6 系列計 5,552 行、`formal_logic` 4 系列計 96 行、`Yahoo_JAPAN_JNLI_Full` 19,992 行があります。これらの ALU は形式上の 0/1 を計算しますが、自然文からその 0/1 を選ぶ過程を回路は検証しません。日本語と Python の論理を結ぶには、各行の前提・真偽値・結論を別に照合する必要があります。未照合の論理行は受理していません。

`v43_relational_causal_cot` 15,000 行には ALU trace がなく、日本語の因果説明を Python に強制する必要はありません。`GSM8K_Japanese_Full` 7,329 行の実例には、1 個 10 円の商品を 1 万個買って 30% 割引なのに、答えが 700 円という不一致がありました。この系列も、元の CPU 観測が一致するだけでは受理しません。`JMultiWOZ_Dialogue_State_Tracking` 4,331 行は全行で元のスロット値が語頭1字に切られていました。会話と Python の良質な対応教材には加えません。原行・生成ソースの証拠は [`logic_and_dialogue_readonly.md`](logic_and_dialogue_readonly.md) に保存しています。
