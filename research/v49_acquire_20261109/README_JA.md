# v49 外部データ6種の取得・検査パイプライン

Hugging Face の公開データ6種を、**Google Colab の CPU ランタイム**で Google Drive に取得し、検査して、GeoCPU の学習用の行（JSONL）に直します。
パソコンには何も取ってきません。GPU も使いません。パスワード・API キー・トークンも要りません（6つとも公開・承認不要のリポジトリです）。

## Colab での開き方

次の URL を開くと、GitHub 上のノートブックが Colab で開きます。

https://colab.research.google.com/github/corcondor/geocpu-diagnostics/blob/research/v49-acquire-20261109/research/v49_acquire_20261109/colab_acquire_check.ipynb

## 実行の手順

1. 「ランタイム → ランタイムのタイプを変更」で、ハードウェア アクセラレータが **なし（CPU）** になっていることを確かめます。
2. 「ランタイム → すべてのセルを実行」。セル3a で Drive のアクセス許可を聞かれるので許可します。
3. 各セルの最後に `PASS` か `FAIL` が出ます。`FAIL` のときはそこで止まります。メッセージの内容を直してから、もう一度「すべてのセルを実行」します。取得済みのファイルは検証して飛ばし、変換済みの行は Drive から戻すので、続きから動きます。
4. 7a（最初の2,000件だけの試運転）の最後に、全体の所要時間の目安が出ます。これは見積もりで、実測ではありません。コードの実行検証（coding）が一番長くなります。
5. 最後のセル10に `REPORT_JA.md` の中身が出ます。

Colab 側の目安: メモリ 12GB、ディスク約100GB のうち、この処理は約4GB（取得したファイルの写し＋変換結果）を使います。

## セルの一覧

| セル | 内容 |
|---|---|
| 1 | 説明 |
| 2 | `pip install -q huggingface_hub pyarrow`（PyPI のみ） |
| 3a | Drive をつなぐ。`DRIVE_ROOT`、`RUN_SUFFIX`、`STOP_ON_COUNT_MISMATCH`、`ONLY` を決める |
| 3b | acquire.py / checks.py / convert.py / pipeline.py と、v48 の評価セット4種（各128行）を `/content/v49` に書き出して読み込む |
| 4 | 試運転（通信なし・書き込みなし）。取得予定の合計バイト数（約1.92GB）を表示 |
| 5 | 6つを取得（再開できる）。ライセンスを全部確認してからデータを取る。コミット固定、大きさと sha256 の照合 |
| 6a | train のファイルだけを Colab のディスクに写して sha256 を照合 |
| 6b | モデルの語彙（3,898件）と評価データを確かめ、13文字の断片の索引を作る |
| 6c | 元データの検査: 件数をカードと照合、語彙外の文字、長さ、目視用の100行 |
| 7a | 変換の試運転（各データセットの最初の2,000件） |
| 7b | 変換の本番 |
| 8 | 絞り込み: 完全一致、近い重複（MinHash）、評価データとの13文字一致、旧コーパスとの重複 |
| 9 | 最終の行、`registry.json`、`REPORT_JA.md` を書く |
| 10 | 報告を表示して、Drive への書き込みを終わらせる |

## どこに何が書かれるか

すべて `GeoCPU_v47_fullmodel_20261106/` の下の**新しいフォルダ**です。既にあるファイルを上書きも削除もしません。同じ名前で内容の違うファイルが既にあれば、そこで止まります（`RUN_SUFFIX` を `'_2'` などにすると、新しいフォルダに書けます）。

```
external_data_v49/                          取得した元ファイル
  <リポジトリ名の / を __ にしたもの>/
    README.md                               データカード（ライセンスの文言はここ）
    <リポジトリ内のパス>                     train のファイルだけ
    provenance.json / PROVENANCE.md         コミット・sha256・ライセンス・帰属表示の記録
  do_not_train_dev/…                        JaQuAD の dev（照合専用。学習には使わない）
  do_not_train_validation/…                 HotpotQA の validation（照合専用）
external_data_v49_rows/<データセット>/       最終の行（採用候補）
  train.jsonl  dev.jsonl  heldout.jsonl  manifest.json  ATTRIBUTION.md
v49_acquire_20261109/
  registry.json                             機械が読む記録
  REPORT_JA.md                              人が読む報告（日本語の表）
  reading/raw_<データセット>_100.txt        元データの100行（目視用）
  reading/final_<データセット>_100.txt      最終の行の100行（目視用）
  converted_unfiltered/<データセット>/       絞り込み前の変換結果（再開用）
  filters/<データセット>_dropped.tsv        落とした行の一覧（行番号・id・理由）
  filters/semantic_review.txt               評価問題に似た行の目視用リスト
  state/*.json                              各セルの結果（再開用）、vocab_cache.json
  run_info_<日時>.json                      実行環境と各セルの所要時間
```

データセット名（フォルダ名）: `coding`（llm-jp/Synthetic-JP-EN-Coding-Dataset）、`tengentoppa`（DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k）、`jaquad`（SkelterLabsInc/JaQuAD）、`hotpot`（hotpotqa/hotpot_qa、distractor）、`toolace`（Team-ACE/ToolACE）、`wizard`（llm-jp/wizardlm8x22b-logical-math-coding-sft-ja）。

行の形は既存のファミリーと同じです（`id, source, license, split, training_category, family, structural_template, difficulty, _v46_group, prompt, target, full_text, loss_mask_spans, tool_steps, reference_final_answer, answer_type`。加えて `attribution` と `adoption`）。`full_text` は `prompt + target` と完全に同じで、`len(full_text)+1` は 2,048 以下（実際は 1,900 文字以内）です。ツールの出力（Python の実行結果、関数の結果）は `loss_mask_spans` と `tool_steps` で損失から外してあります。分割は文章の単位（グループ）で、heldout 3%・dev 1%・train 96% です。

## REPORT_JA.md の読み方

- **1. 結果の一覧**: データセットごとに、元の件数 → 変換後 → 完全重複除去後 → 近似重複除去後 → 評価データ除外後 → 旧コーパス重複除去後（これが採用候補の行数）。各段階は前の段階に残った行だけを対象にした順番の絞り込みです。教師あり文字数は、`target` から損失に入らない部分（ツールの出力）を引いた文字数です。
- **採用判定は暫定の提案**です。事前に決めた件数はなく、検査の結果から出ます。種類は次のとおりです。
  - 採用候補: 人手で作られたデータ（jaquad、hotpot）と、コードを実行して確かめた行を含む coding。
  - 条件付き候補: tengentoppa（Qwen2.5-32B が作り、人手の確認がない）。
  - 低優先の採用候補 / 保留: wizard。算術式の誤り率が基準（既定: 誤り率 1% 以下で、300式以上を確認できていること）を満たせば低優先の候補、そうでなければ保留。この基準は私の仮の値で、決めるのはあなたです。
  - 保留: toolace（生成モデルがカードにも論文にも書かれていないため、確認できるまで保留）。
  - 不採用: 採用できる行が0件のとき。
  - どの判定も、各データセットの目視100行（`reading/` のファイル）を読むまでは確定しません。保留のデータセットも行ファイルは作ってありますが、`adoption` の欄に保留と書いてあります。あなたが決めるまで学習に混ぜないでください。
- **2. 全体の確認結果**: 旧コーパスで確認できた行数（423,348 のはず）、語彙の出どころ、使った評価データの一覧、評価データとの一致で落とした行数。
- **3. 件数の不一致（MISMATCH）**: 数えた件数がカードや計画の数字と違うデータセット。カードの数字が違っている可能性があります。
- **4. 詳細**: 変換での主な除外理由、語彙外文字の割合、2,048文字を超える行の割合、検証の内訳、wizard の算術式の誤り率など。
- **5. 検査の範囲と限界**: 見ていないこと（GSM8K・MATH のテスト問題は Drive に無く、承認された6つ以外は取ってこないため、照合に入っていません。入れたいときは `v49_eval_extra/` に .jsonl か .json で置いて、`RUN_SUFFIX` を変えて再実行します）。13文字の一致は言い換えや翻訳を見つけられません。
- **6. ライセンスと表示の義務**、**7. 出力の場所**。

## ライセンスと表示の義務

- 4つ（coding、tengentoppa、toolace、wizard）は **Apache-2.0**（カードの記載）。ライセンスの文言と著作権表示を保ち、変更したこと（JSONL の行に変換し、絞り込んだ）を明記します。出典の文言は各 `external_data_v49_rows/<データセット>/ATTRIBUTION.md` にあり、各行の `source` と `license` の欄にも入っています。
- **JaQuAD は CC BY-SA 3.0**、**HotpotQA は CC BY-SA 4.0** です。JaQuAD は日本語の、HotpotQA は英語の Wikipedia の文章を含みます。出典の表示、変更の明記、そして **ShareAlike（継承）** が必要です。変換した行を公開・再配布するときは、同じ CC BY-SA（または互換のライセンス）にします。行の `license` の欄は `CC-BY-SA-3.0` / `CC-BY-SA-4.0` です。学習したモデルの重みが「翻案」にあたるかどうかは法律の判断になり、このパイプラインは判断していません。公開する前に確かめてください。
- 文章を作ったモデルの利用条件（Qwen2.5-32B、Nemotron-4-340B、WizardLM-2 など）はここでは確認していません。各データセットのカードのライセンス表記だけを根拠にしています。
- ToolACE のライセンス（Apache-2.0）はカードの記載です。生成モデルは未確認です。

## 守っていること（オーナーの決定）

- 重い処理はすべて Colab。パソコンには取ってこない。
- Drive には新しいフォルダだけ。上書きも削除もしない。内容が違えば止まる。
- 通信先は huggingface.co（と配信元）と PyPI だけ。コミットを固定し、大きさと sha256 をハブの値と照合する。
- 評価データ・dev・validation・test では学習しない。JaQuAD の dev と HotpotQA の validation は、照合専用の別フォルダに置く。
- ツールの出力は損失から外す（`loss_mask_spans`、`tool_steps`）。
- 旧コーパス 423,348 行と v48 の 4,096 行には触らない。旧コーパスは読むだけで、重複の確認に使う。

## 手元（パソコン）でのテスト

どれも小さな見本（合計 100KB 未満）だけを使います。データセットは取ってきません。1つずつ実行します。

```
python -I -X utf8 test_acquire.py
python -I -X utf8 test_checks.py
python -I -X utf8 test_convert.py
python -I -X utf8 test_pipeline.py       # ノートブックの各セルの処理を、見本で最初から最後まで動かす
python -I -X utf8 test_make_notebook.py  # ノートブックの形・埋め込んだコードが元と同じこと・GPU も秘密も無いこと
```

ノートブックを作り直すとき: `python make_notebook.py`（`colab_acquire_check.ipynb` ができます）。コードを直したら、必ずこれを実行してから commit します（テストが埋め込みの一致を確かめます）。
