# 既存評価を変更せず比較するための凍結レジストリ

`frozen/registry.json` に問題数・固有prompt数・familyキー・正式mode・順序付き選択ハッシュ・生成条件を保存しています。`frozen/source_files.json` は取得した原本のSHA-256とZIP内の出所です。元の評価問題、採点ソース、runnerはbyte単位で保存し、追加の大規模問題はこのレジストリに混ぜません。**この成果物は入力とソースの検証であり、モデル性能の実測結果ではありません。**

## 件数と区別

| 既存評価 | 原本の正式な範囲 |
|---|---|
| T1 / T2 | 各200行。T2は固有prompt193 |
| T3 JamC-QA | logit 2,309問、原文形式の自由生成は先頭100問 |
| T4 | 8,100行（JMMLU 7,536、コード状態追跡164、ARC 400）。固有prompt8,079 |
| T5 | 400行、100文脈×4 operator_type。原表示名の「6-Operator」は書き換えない |
| T6 | 200行、固有prompt21。2-hopラベル100行の固有promptは4 |
| T7 | 元のMC抽出後の先頭200問をclosed/open各一回。生成予算は元の60/120 |
| T8 | 4問題×元形式／整列形式。各4件を別集計。数学正解率ではなく形式評価 |
| Python120 | 日本語80＋Numina40、120 family。原選択・原採点を維持 |
| extended74 | engineering **64**＋visual **10**。engineering単独74ではない |
| search decision / final | 200／198。次クエリ選択と、最終根拠入力後の回答・停止を別評価 |
| public dev80 | 検索深度0〜3を各20。原抽出順と採点を維持 |
| new-public 9群 | 原設定raw最大12／NLL最大64。実データ3行の群は3件。NLLは正解率ではない |

行数、文字列の固有数、実データにあるfamilyキー数は別の値です。固有prompt数だけで問題の独立性を保証しません。重複問題の削除・再配点は行っていません。

## Colab runner向けAPI

```python
import frozen_eval as frozen

frozen.verify_frozen()                    # CPU、ファイル読取のみ
cases = list(frozen.iter_cases("python120"))
kwargs = frozen.generation_kwargs("python120")
scorer = frozen.load_original_scorer("python120")
# generation = model.generate_v44fix(case["prompt"], vocab, inverse, **kwargs)
# result = scorer.score_generation(case, generation)
```

`engineering64` は元 `score_case(case, generation)`、`visual10` は元 `evaluate(case, model_generated_text, artifact_dir)` を返します。検索2種は元main内の採点文をASTのまま再利用する `score_search_decision(case, generation)`／`score_search_final(case, generation)` です。生成部分を除き、正規化・終了条件・採点式を変更していません。採点例外は呼出元へ伝え、未測定を記録できるようにしています。例外を0点や正解に置き換えないでください。

`iter_heldout_prompts()` は原本manifest、選択されなかった公開dev全件、T8両形式、別欄にある自然質問・実際に構成されるT3/T7入力も返します。戻り値の `prompt`／`prompt_sha256`／`families` を追加学習データの漏洩検査に使えます。完全一致だけでは言い換えや意味的同一問題を除外しきれません。

## T1〜T8を原stageで実行

`build_official_tier_command(study=..., run=..., step=..., checkpoint_sha=..., output=...)` は無変更の `stage_official_tier_v47.py` 用argvを作るだけで、実行や学習はしません。runner側で実際に検証したcheckpointのSHAと新規出力先を渡してください。

```bash
python frozen_eval.py verify
python frozen/continue/tier_stage/stage_official_tier_v47.py \
  --study /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106 \
  --run SAVED_RUN --step SAVED_STEP \
  --checkpoint-sha ACTUAL_64_HEX_SHA --output NEW_RESULT_DIRECTORY
```

正式stageはT1〜T7を元の `IntegratedModelV42`、T8を元のv44fixで実行します。新共通v44fix生成器に置換すると元の比較条件が変わります。stageはstudy内の元bundleと `v47_continue25500_training_v1.zip`、checkpoint／receipt／training_saved／input_manifest整合を検査します。原コードは中断済みTierログの自動上書きを拒否します。

元のPython120／extendedの30500向けCLIはstep・branchが固定です。未来checkpointは、新runnerが独立に保存済みcheckpointを検証したうえで、固定caseと原scorer APIを使います。`launch_suite_30500.py` は学習も起動するため評価入口には使いません。

### 完了済みT1〜T8を再生成せず利用する

`eval_candidate.py run ... --reuse-original-from ORIGINAL_TIER_DIRECTORY` は、元の `input_binding.json` とcheckpoint／学習入力／元コードの対応、`complete.json` の全log/rawハッシュ、stage完了のreturncode=0、T4の8,100 ID・target、T8両形式の4 IDを読み取り検査します。成功時に原票15ファイルだけを新しい評価出力内へコピーし、別の `reuse_receipt.json` を作ります。**元の原票・ロック・学習ファイルには書き込みません。** 元側にstage lockがある場合は再利用を拒否します。

元stageの `command_sha256` は出力パスに依存します。コピー後も値を改変せず保持するため、**コピー先にnative stageを再実行しないでください**。新runnerはコピーを自分のreceiptと全ハッシュで検証し、`--resume` 時もnative stageを起動しません。既存baselineの正式T1〜T8を重複生成せず、別の補助評価を進められます。

補助runnerはcase ID・全case内容・checkpoint／scorer／生成条件を結び付け、完了rawのハッシュも再確認します。保存済み生成があり採点前に中断した場合は、その生成を再利用します。新しいsuiteの `--resume` は新規loggerとして開始できます。raw JSONではtensorを持つ `cpu_state` だけを除外し、元の `cpu_trace`、ツール履歴、モデル生成文を残します。生成器・設定・記録先の障害は停止し、連続する採点障害でも全問を消費し続けません。

新しい知識診断は `knowledge_free_dev.jsonl` の64 chainをclosed／tool／oracleで比較します。同relationの誤答候補だけで作った小さいMC devは別ファイルに維持し、既定の知識診断には使いません。正式T3/T4の問題は変わりません。

既存stageの進捗表示はT1=50件、T2=20件、T3=logit512／生成25件、T4=logit1024／生成100件、T5/T6=40件、T7=20件ごとです。外側のrunnerでstdout/stderrを即時転送し、待機中heartbeatを表示できます。新補助runnerは一問ごとに記録します。

## 再現範囲と未測定

旧 `gate_v44b` と `sentinel_v46` は採点ソースを保存していますが、`train_v43_knowledge_unified.jsonl`、およびgate用 `bridge_heldout.jsonl` が取得bundleにありません。これらは入力不足として登録し、完全再現・合格を報告しません。trainingの5ファイルは元の学習契約を参照するため保存してありますが、このAPIは起動しません。

CPU検証は `python -m unittest discover -s tests -p test_frozen_eval.py -v`。原本ハッシュ、正式modeの件数・順序、改変拒否、検索・Pythonで観測文だけを最終回答扱いしないこと、全選択promptの漏洩検査集合への包含を確認します。GPU生成やcheckpoint精度検証は実行していません。

`tests/test_eval_candidate.py` は、CPU上のmock生成・仮のcheckpoint bytesで、再開、原票コピー元の不変、誤ったrawの拒否、native metric、例外の分離、完了状態を検証します。これらのmock正誤はモデルの性能値ではありません。実モデルのロード、GPU上の実行、Colab環境での依存解決は未検証です。
