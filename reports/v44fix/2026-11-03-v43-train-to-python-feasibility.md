# v43 の実教材を純 Python v44 に継承できる範囲

元の v43 ZIP を読み取って調査した。学習、生成、重み読込、教材変換は実行していない。集計コードは [`audit_v43_train_conversion_scope.py`](../../src/pure_v44_candidates/audit_v43_train_conversion_scope.py)、全数の小さな[集計原票](v43_train_conversion_counts.json)は SHA256 `9c2a03f0cb7d27f576ee306f1f05bfdec9d097079628411fc4d4349effd8eb10`。対象は `G:/マイドライブ/GeoCPU_Frontier_CoTraining_v43/Grand_cotraining_v43.zip` 内 `data/train_v43_knowledge_unified.jsonl`。同ZIPにある評価ファイルだけを重複照合した。

## 実際に残っている教材

268,032行、SHA256で数えた固有のprompt/target組は224,113。文字列検索で `<SEARCH ` を含む行は29,000、CPU命令候補の正規表現に当たる行は133,896、Python code fenceを含む行は0。この133,896は実行可能なCPU行の認証数ではない。主なsourceは `SoftBank_JSQuAD_Full` 61,954、`Algebraic_Composite_Aligned` 9,828、`GSM8K_Japanese_Full` 7,329、検索強化の3 sourceがそれぞれ4,000、3,000、10,000、学術検索CoTが12,000行など。旧教材には日本語読解・推論も残っている。

実例：`Algebraic_Composite_Aligned` の問いは `(93 * 12 + 15) % 11` を求める元prompt `問題: …\n思考:`。targetは日本語説明の中で `<MUL R0 93 12> => [R0=1116]`、`<ADD R0 R0 15> => [R0=1131]`、`<MOD R0 R0 11> => [R0=9]` を挟み、最後に `答え: 9` とする。`swe_alu_accumulate_sum` の元promptには既に Python コード `R0 = 73; for i in range(1): R0 += i * 2` があり、targetは `<SET>/<ADD>/<DEC>` の状態列と最終値73を含む。`search_augmented_1hop_fact` は `<SEARCH 憲法 9条> => [RESULT: …]` と説明を含み、2hop sourceは2回の検索結果をつないでいる。これらを単純に削除すれば、旧能力を支えた問い・行動・観測・結論の対応が教材から失われる。

純v44の現行作成器は[`prepare_public_v44.py`](<C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/prepare_public_v44.py:256>) 263–266、396–397行で旧ALU/SEARCH行をアンカーから拒否する。一方、純v44の[`tir_engine_v44r1.py`](<C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/tir_engine_v44r1.py:150>) は**行頭**の ` ```python` だけ実行し、` ```output` 観測を注入する。同workerの[`executor_worker_v44r1.py`](<C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/executor_worker_v44r1.py:81>) は `search(query)` を提供するが、offline callbackが必要。旧CPUのインライン `<ADD ...> => ...` を置換するだけでは実行されない。

## 安全に変換する条件と限界

- **計算の一部は変換可能。** 元の演算と日本語結論を保ち、実際に実行されるPython code fenceへ変える。例えば上記算数なら `R0=93*12; R0+=15; R0%=11; print(R0)` の実出力9と旧結論9を一致検査できる。SWEのように多命令を含む行は1命令1ツール呼出しでは現行上限8回を超え得るため、ひとまとまりのPython計算にする必要がある。元の`qwenputer_controller_v30.py` 126–176行はレジスタを8本に丸め、数値を32bit化し、ゼロ除算の除数を1へ変える。Pythonの無限精度整数や通常のゼロ除算と同一視せず、元回路の実行結果との一致を行ごとに検査する。**実train全行の変換成功率は未測定。**
- **検索は一括変換できない。** 29,000行の旧検索のうち、7,000行は合成された `[RESULT: …]`、他には外部検索のスニペットを含む行がある。純v44 workerの`search(query)`は使えるが、同梱 `knowledge_corpus_v43.jsonl` は15文書。旧teacherの検索文面が現在のoffline検索結果と一致するとは限らない。Pythonで旧結果を文字列として`print`するだけでは、検索を実行した学習とは呼べない。観測を実callbackの出力で再生成し、根拠・最終結論が一致する行だけを受理するか、検索を教えない通常の知識回答として別管理する必要がある。
- **入力形式を保つ。** 旧行のpromptは `問題/質問 …\n思考:`、target先頭に空白がある。純v44公開3群の `ユーザー …\nアシスタント:` へ機械的に変えると、旧重みの既知入力を失う。この調査では旧promptのバイト列を保つことを変換の条件とする。形式を統一するなら別介入として検証する。
- **教師ラベルを分ける。** 純v44[`training_data_v44.py`](<C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/training_data_v44.py:7>) 16–29、62–72行の現行実装は、モデルが出す説明・コードを正解位置、環境が注入する` ```output`を`target_segments`のtool/train=false、真EOSを正解位置として扱える。変換行でこの区別を作り、`full_text=prompt+target`、全語彙合法、回答＋EOSが8,192位置以内という入口を維持する。教師の観測をモデル出力として学ばせる変換は不可。

## 評価の分離

同ZIPに同梱の評価・manifestのprompt総数11,388とtrainを**完全一致**で比較すると、7,000 train行が `eval_v42_search_retrieval_aligned.jsonl` の200 promptと重なる。内訳は1hop 4,000と2hop 3,000。他の同梱 `eval_v42_search_retrieval_heldout`、SWE held-out、v33/v38、JamC-QA、8,100問manifest、論理held-outとの完全一致は0。したがって aligned200を未見評価に用いないか、変換対象から一致7,000行を除く。held-outとの完全一致0は、言い換え・同一family・共有数値の漏洩0を意味しない。変換時は全評価/heldのID、元prompt、正規化prompt、生成familyを再照合する。

**結論：** 旧ALU教材にはPython TIRへ意味を移せる行があるが、元回路との数値一致と実注入観測を検証して初めて教材として受理できる。旧検索29,000行の一括移植は、現行15文書のoffline検索だけでは根拠不足。今回の確認は変換の設計可能性であり、変換済み教材・学習効果の実証ではない。
