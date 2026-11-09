# v49 外部データ6種の取得・検査パイプライン

Hugging Face の公開データ6種を、**Google Colab の CPU ランタイム**で Google Drive に取得し、検査して、GeoCPU の学習用の行（JSONL）に直します。
パソコンには何も取ってきません。GPU も使いません。パスワード・API キー・トークンも要りません（6つとも公開・承認不要のリポジトリです）。

## Colab での開き方

次の URL を開くと、GitHub 上のノートブックが Colab で開きます。ブランチ名に `/` が入っているので、開けないときは `/` を `%2F` にした2つ目の URL を試してください。

https://colab.research.google.com/github/corcondor/geocpu-diagnostics/blob/research/v49-acquire-20261109/research/v49_acquire_20261109/colab_acquire_check.ipynb

https://colab.research.google.com/github/corcondor/geocpu-diagnostics/blob/research%2Fv49-acquire-20261109/research/v49_acquire_20261109/colab_acquire_check.ipynb

どちらも開けないとき（ブランチをまだ push していない場合など）は、`colab_acquire_check.ipynb` を手元に置いて、Colab の「ファイル > ノートブックをアップロード」（File > Upload notebook）で開きます。ノートブックは自分の中に4つのコードを抱えているので、これ1つで動きます。

## 実行の手順

1. 「ランタイム → ランタイムのタイプを変更」で、ハードウェア アクセラレータが **なし（CPU）** になっていることを確かめます。GPU が付いたままだとセル3a が止まります（有料の計算資源を使わないため。A100 のセッションの使い回しを防ぎます）。
2. 「ランタイム → すべてのセルを実行」。セル3a で Drive のアクセス許可を聞かれるので許可します。コードを実行するあいだ Drive を外してつなぎ直すので（下の「サンドボックス」）、変換のあいだに許可の画面がもう一度出ることがあります。
3. 各セルの最後に `PASS` か `FAIL` が出ます。`FAIL` のときはそこで止まります。メッセージの内容を直してから、もう一度「すべてのセルを実行」します。取得済みのファイルは検証して飛ばし、変換済みの行・シャードは Drive から戻すので、続きから動きます。
4. 7a（最初の2,000件だけの試運転）の最後に、全体の所要時間の目安が出ます。これは見積もりで、実測ではありません。コードの実行検証（coding）が一番長くなります。
5. 7b は長くかかります。**ブラウザのタブを開いたまま**にしてください（閉じると実行中のセルが止まることがあります）。止まっても、できたシャード・データセットは Drive にあります。実行中のセルを自分で止めたときは、動いている変換プロセスも一緒に止まります。
6. 最後のセル10に `REPORT_JA.md` の中身が出ます。

Colab 側の目安: メモリ 12GB、ディスク約100GB のうち、この処理はローカルに約4GB（取得したファイルの写し＋変換結果）を使います。Drive には元ファイル約1.9GB に加えて、変換済みの行と最終の行が入ります（7a が見積もりを出し、足りなければ 7b の前に止まります）。

## セルの一覧

| セル | 内容 |
|---|---|
| 1 | 説明 |
| 2 | `pip install -q "huggingface_hub>=0.24" pyarrow`（PyPI のみ。入った版を表示。読み込み済みと違えばセッションの再起動を促して止まる） |
| 3a | GPU が付いていたら止まる。Drive をつなぐ。`DRIVE_ROOT`、`RUN_SUFFIX`、`RAW_SUFFIX`、`STOP_ON_COUNT_MISMATCH`、`ALLOW_VOCAB_MISMATCH`、`ONLY`、`UNMOUNT_DRIVE_DURING_CODE_RUNS`、`FLUSH_DRIVE_AFTER_STEPS` を決める |
| 3b | acquire.py / checks.py / convert.py / pipeline.py と、v48 の評価セット4種（各128行）を `/content/v49` に書き出して読み込む（同じカーネルの古い版は捨て、読み込んだ中身が埋め込みと同じことを確かめる） |
| 4 | 試運転（通信なし・書き込みなし）。取得予定の合計バイト数（約1.92GB）と Drive の空きを表示。前の中断で残った `.stale-N` ファイルも一覧にする（消すのはあなたの手で） |
| 5 | 6つを取得（再開できる）。ライセンスを全部確認してからデータを取る。以前の実行がコミットを記録していれば、そのコミットを取る。大きさと sha256 の照合。途中まで写した `.part` は、先頭が一致すれば続きから写す |
| 6a | train のファイルだけを Colab のディスクに写して sha256 を照合 |
| 6b | モデルの語彙（3,898件）と評価データを確かめ、13文字の断片の索引を作る。語彙の文字集合がパソコン上のものと違えば止まる（`ALLOW_VOCAB_MISMATCH`）。必須の評価ファイルが1つでも13文字の断片を出さなければ止まる。T1〜T8 の対応表を出す |
| 6c | 元データの検査: 件数をカードと照合、語彙外の文字、長さ、目視用の100行。壊れた行や読めないバイトは数えて大きく表示（止まらない） |
| 7-0 | サンドボックスの自己診断。データ中のコードを実行する前に必ず通る |
| 7a | 変換の試運転（各データセットの最初の2,000件）。Drive は外れている |
| 7b | 変換の本番。coding と tengentoppa は小分け（シャード、既定20,000件ずつ）にして、できた分から Drive に保存。Drive は変換の波ごとに外れる |
| 8 | 絞り込み: 完全一致、近い重複（MinHash）、評価データとの13文字一致、旧コーパスとの重複、dev / heldout の近い兄弟。**ここまでの結果は先に Drive に書く** |
| 8b | 任意の確認（評価問題に似た行の目視用リスト、データセット間の重複）。それぞれ約15分の上限つき。行は除かない |
| 9 | 最終の行、`registry.json`、`REPORT_JA.md` を書き、Drive をつなぎ直してから全ファイルの sha256 を確かめる |
| 10 | 報告を表示して、Drive への書き込みを終わらせる |

## どこに何が書かれるか

すべて `GeoCPU_v47_fullmodel_20261106/` の下の**新しいフォルダ**です。既にあるファイルを上書きも削除もしません。同じ名前で内容の違うファイルが既にあれば、そこで止まります。書き込みはまず `<名前>.part` に書いて、行き先がまだ無いことを確かめてから名前を付け替えるので、実行が途中で止まっても欠けたファイルは残りません（残った `.part` は `.part.stale-N` に退避するだけで、消しません）。

```
external_data_v49/                          取得した元ファイル（RAW_SUFFIX で別のフォルダにできる）
  <リポジトリ名の / を __ にしたもの>/
    README.md                               データカード（ライセンスの文言はここ）
    <リポジトリ内のパス>                     train のファイルだけ
    provenance.json / PROVENANCE.md         コミット・sha256・ライセンス・帰属表示の記録
  do_not_train_dev/…                        JaQuAD の dev（照合専用。学習には使わない）
  do_not_train_validation/…                 HotpotQA の validation（照合専用）
external_data_v49_stageA/<変換器タグ>/<データセット>/   絞り込み前の変換結果（再開用。RUN_SUFFIX に左右されない）
  <データセット>__shards/<範囲>/             coding・tengentoppa の小分け（シャード）
external_data_v49_rows/<データセット>/       最終の行（採用候補: coding, tengentoppa, jaquad, hotpot）
  train.jsonl  dev.jsonl  heldout.jsonl  manifest.json  ATTRIBUTION.md  LICENSE_NOTICE.txt  [LICENSE-Apache-2.0.txt]
external_data_v49_rows/_HOLD_do_not_mix/<データセット>/   保留（toolace, wizard）。DO_NOT_TRAIN_UNTIL_APPROVED.txt つき。学習に混ぜない
v49_acquire_20261109/
  registry.json                             機械が読む記録
  REPORT_JA.md                              人が読む報告（日本語の表）
  reading/raw_<データセット>_100.txt        元データの100行（目視用）
  reading/final_<データセット>_100.txt      最終の行の100行（目視用）
  filters/<データセット>_dropped.tsv        落とした行の一覧（行番号・id・理由）
  filters/semantic_review_<ハッシュ>.txt    評価問題に似た行の目視用リスト（8b）
  state/*.json                              各セルの結果（再開用）、vocab_cache.json
  run_info_<日時>_<pid>_<n>.json            実行環境と各セルの所要時間
```

変換済みの行（`external_data_v49_stageA/`）は、変換のコード（convert.py・サンドボックス・記号の正規化・語彙）が同じなら使い回されます。絞り込み側（pipeline.py、checks.py の重複・照合の部分）を直したり、`RUN_SUFFIX` を変えたりしても、数時間かかる変換はやり直しません。

データセット名（フォルダ名）: `coding`（llm-jp/Synthetic-JP-EN-Coding-Dataset）、`tengentoppa`（DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k）、`jaquad`（SkelterLabsInc/JaQuAD）、`hotpot`（hotpotqa/hotpot_qa、distractor）、`toolace`（Team-ACE/ToolACE）、`wizard`（llm-jp/wizardlm8x22b-logical-math-coding-sft-ja）。

行の形は既存のファミリーと同じです（`id, source, license, split, training_category, family, structural_template, difficulty, _v46_group, prompt, target, full_text, loss_mask_spans, tool_steps, reference_final_answer, answer_type`。加えて `attribution` と `adoption`）。`full_text` は `prompt + target` と完全に同じで、`len(full_text)+1` は 2,048 以下（実際は 1,900 文字以内）です。ツールの出力（Python の実行結果、関数の結果）は `loss_mask_spans` と `tool_steps` で損失から外してあります。

### 分割（heldout 3%・dev 1%・train 96%）の単位

分割の単位はデータセットごとに違います。**dev / heldout が学習行から独立しているのは、JaQuAD と HotpotQA の2つだけです。**

- **jaquad**: 記事（タイトル）単位。同じ記事の質問は同じ分割に入ります。
- **hotpot**: 根拠の段落が共通する質問を1つに繋いだ連結成分（全 train レコードに対する union-find）単位。根拠の段落を共有する質問は同じ分割に入ります。
- **coding・tengentoppa・toolace・wizard**: 行単位です。これらは機械が作った文で自然なグループがないので、同じ種から作られた兄弟（Evol-Instruct の連鎖など）が train と dev / heldout にまたがって入りえます。8 で、学習行と推定 Jaccard 0.5 以上で似ている dev / heldout の行は落としますが、言い換えレベルの兄弟は残ります。そのため、これら4つの dev / heldout は**グループ独立の評価として使わないでください**（形式の確認用です）。manifest・registry・REPORT_JA.md のどれにも `split_unit` と `group_independent_dev_heldout` が書いてあります。

## REPORT_JA.md の読み方

- **1. 結果の一覧**: データセットごとに、元の件数 → 変換後 → 完全重複除去後 → 近似重複除去後 → 評価データ除外後 → 旧コーパス重複除去後 → dev/heldout の近い兄弟除去後（これが採用候補の行数）。各段階は前の段階に残った行だけを対象にした順番の絞り込みです。教師あり文字数は、`target` から損失に入らない部分（ツールの出力）を引いた文字数です。
- **採用判定は暫定の提案**です。事前に決めた件数はなく、検査の結果から出ます。種類は次のとおりです。
  - 採用候補: 人手で作られたデータ（jaquad、hotpot）と、coding。coding の python 行は、**実行して、答え自身の保存出力と一致したものだけ**が入ります（それが「検証済み」の python 行で、報告の数字はこれだけです）。保存出力が無くて実行しかできなかった行は、数だけ記録して残しません。他言語の code_text 行は実行しておらず、未検証です。
  - 条件付き候補: tengentoppa（Qwen2.5-32B が作り、人手の確認がない）。
  - 保留: wizard（計画の「数学の明らかな誤りがあるものは避ける」に該当）。測れた算術式の誤り率は書きますが、自動では採用候補に上げません。誤り率が小さくても、100行を読んで、あなたが決めます。
  - 保留: toolace（生成モデルがカードにも論文にも書かれていないため、確認できるまで保留）。
  - 不採用: 採用できる行が0件のとき。
  - どの判定も、各データセットの目視100行（`reading/` のファイル）を読むまでは確定しません。保留のデータセットの行は `external_data_v49_rows/_HOLD_do_not_mix/` にあり、各行の `adoption` の欄にも保留と書いてあります。あなたが決めるまで学習に混ぜないでください。
- **2. 全体の確認結果**: 旧コーパスで確認できた行数（423,348 のはず）、語彙の出どころ、使った評価データの一覧と T1〜T8 の対応表、評価データとの一致で落とした行数、dev / heldout の近い兄弟の除去数。
- **3. 件数の不一致（MISMATCH）**: 数えた件数がカードや計画の数字と違うデータセット。カードの数字が違っている可能性があります。
- **4. 詳細**: 変換での主な除外理由、語彙外文字の割合、2,048文字を超える行の割合、検証の内訳、英語の行の数と教師あり文字数、近似重複の群の大きさの分布、wizard の算術式の誤り率など。
- **5. 検査の範囲と限界**: 見ていないこと（下の「していないこと」）。
- **6. ライセンスと表示の義務**、**7. 出力の場所**。

## サンドボックス（データ中のコードを実行する仕組み）の正直な説明

coding の python 行は、データ中のコード（LLM が書いたもの）を実行して、保存された出力を実行結果で置き換えます。

- 規則は **モデル自身の実行器と同じ**です（`checks.static_check(profile="executor")`）。使える import は math・cmath・decimal・fractions・statistics・itertools・functools・operator・collections だけで、クラス定義、属性への代入、アンダースコアで始まる属性、`open`・`eval`・`exec`・`getattr`、`import re` / `json` / `random` などは使えません。実行器が受け付けないコードを、「成功した観測」の行にして学習させることはありません。以前の広い規則（`wide`）は廃止しました。
- さらに、レビューで見つかった抜け道（`from operator import attrgetter, methodcaller`、組み立てた文字列の `.format`、ジェネレータのフレーム、`random._os` のようなモジュール経由の到達）を静的に拒否し、モジュールは「公開された、モジュール以外の属性」だけの名前空間で渡します。
- ただし、**Python の中だけの制限は壁ではなく、ふるいです**。壁は OS です。Colab では、子プロセスは **`nobody`** という権限のないユーザで動き（補助グループなし）、CPU 時間・メモリ・ファイルサイズ・プロセス数・ファイル数に上限があり、出力はファイルに書かせて末尾だけを読みます（パイプで巨大な出力を溜めない）。タイムアウトでは子プロセスの属するセッションごと止めます。
- **コードを実行しているあいだ、Drive はアンマウントされます**（`UNMOUNT_DRIVE_DURING_CODE_RUNS`、既定でオン）。万一コードが抜け出しても、あなたの Drive（重み、オプティマイザ、旧コーパス）には届きません。変換プロセスが触るのはローカルの `/content/v49_work` だけです。
- **7-0 の自己診断**が、この Colab の上で次のことを確かめ、1つでも外れたら止まります（弱い代わりの方法には切り替えません）: 子プロセスが root でない・つながっている Drive を一覧も書き込みもできない・`/etc/shadow` を読めない、上の抜け道のコードが拒否される、許可した9つのモジュールがすべて読み込める（メモリ上限と `-S` のもとで）、無限ループが止まる、出力が上限で切れる。
- 限界: Colab の VM そのものを守る壁ではありません。データは公開された LLM 生成データで、悪意は想定していませんが、「抜け出せないことの証明」ではありません。

## ライセンスと表示の義務

- 4つ（coding、tengentoppa、toolace、wizard）は **Apache-2.0**（カードの記載）。ライセンスの文言と著作権表示を保ち、変更したこと（JSONL の行に変換し、絞り込んだ）を明記します。各 `external_data_v49_rows/<データセット>/` に `ATTRIBUTION.md` と `LICENSE_NOTICE.txt` があり、Apache-2.0 の本文（`LICENSE-Apache-2.0.txt`、Colab に入っている Apache-2.0 のパッケージの LICENSE から写したもの。見つからなければ `LICENSE_NOTICE.txt` にその旨と取得先を書く）も置きます。各行の `source` と `license` の欄にも出典が入っています。
- **JaQuAD は CC BY-SA 3.0**、**HotpotQA は CC BY-SA 4.0** です。JaQuAD は日本語の、HotpotQA は英語の Wikipedia の文章を含みます。出典の表示、変更の明記、そして **ShareAlike（継承）** が必要です。変換した行を公開・再配布するときは、同じ CC BY-SA（または互換のライセンス）にします。**2つを1つの単一ライセンスのファイルに混ぜて再配布しないでください**（BY-SA 3.0 と 4.0 の行は別々のラベルのままにします。行の `license` の欄は `CC-BY-SA-3.0` / `CC-BY-SA-4.0`）。ライセンスの条文へのリンクは `LICENSE_NOTICE.txt` にあります。学習したモデルの重みが「翻案」にあたるかどうかは法律の判断になり、このパイプラインは判断していません。公開する前に確かめてください。
- 文章を作ったモデルの利用条件（Qwen2.5-32B、Nemotron-4-340B、WizardLM-2 など）はここでは確認していません。各データセットのカードのライセンス表記だけを根拠にしています。
- ToolACE のライセンス（Apache-2.0）はカードの記載です。生成モデルは未確認です。

## していないこと（正直な一覧）

- **GSM8K・MATH のテスト問題との照合**: Drive に無く、承認された6つ以外は取ってこないので、照合に入っていません。入れたいときは `v49_eval_extra/` に .jsonl か .json で置いて、`RUN_SUFFIX` を変えて 6b 以降をやり直します（変換済みの行は使い回します）。取ってくること自体は、あなたの許可があれば足せます。
- **登録簿の他の評価セット**（python120 の selection_120.json、engineering64、visual10、new_public の *_dev.jsonl など）: Drive に置いてあれば照合に入ります（単体のファイル、バンドルの中、小さなスイート zip の中を探します）。見つからないものは、報告の 5 に名前を挙げて「照合していない」と書きます。
- 言い換え・翻訳との重複は検出できません（13文字の完全一致と MinHash だけ）。意味的な類似の確認（8b）は文字の2-gramの目視用リストで、網羅的ではなく、行は除きません。
- 近似重複は MinHash の推定です（128置換、推定 Jaccard 0.75 以上を同じ群とする。真の Jaccard 0.85 以上はほぼ確実、0.8 で約93%、0.75 で約55%、0.7 で約13%の確率）。群は連鎖でつながるので、残す行と直接似ていない行が落ちることもあります（群の大きさの分布を報告に出します）。
- 生成モデルの利用条件は未確認です。
- 本物の Colab でも、本物の Hugging Face でも、まだ実行していません。手元の見本（100KB 未満）での単体テストと、偽のハブでの取得テストだけです。JaQuAD の実ファイルの構造、parquet の実際の列、フルサイズでの時間とメモリ、Drive のマウントの挙動は、最初の実行で初めて確かめられます。

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
python -I -X utf8 test_pipeline.py       # ノートブックの各セルの処理を、見本で最初から最後まで動かす（数分かかる）
python -I -X utf8 test_make_notebook.py  # ノートブックの形・埋め込んだコードが元と同じこと・GPU も秘密も無いこと
```

ノートブックを作り直すとき: `python make_notebook.py`（`colab_acquire_check.ipynb` ができます）。コードを直したら、必ずこれを実行してから commit します（テストが埋め込みの一致を確かめます）。
