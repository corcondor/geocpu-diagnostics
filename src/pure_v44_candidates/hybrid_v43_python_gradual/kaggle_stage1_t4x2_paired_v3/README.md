# Kaggle T4 2 枚の stage 1 対照実験

このディレクトリは、v43 の step 15000 を共通の親にして、0% 対照群と 1% Python 群を各 GPU 1 枚で同時に学習する私有 Kaggle 実行用パッケージを作ります。先行する v1 実行は最初の更新で GPU メモリ不足になりました。v2 は小分け学習の位置数を 4,096 に抑えましたが、Kaggle の保存処理が HTTP 400 を返し、ノートブックは作成されませんでした。この v3 は短い名前 `geocpu-v43py-stage1-m4096-20261104` を使います。実効バッチ 64 件、損失の式、教材計画、モデルは同じです。各群は 250 回の重み更新で停止します。次の学習段階への昇格は実装していません。`prepare.py` はパッケージを作るだけで、Kaggle へのアップロードやジョブの起動は行いません。

## 入力と実行条件

- 元の入力データセットは `geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103` です。元の v43 ソース ZIP と step 15000 のチェックポイントのハッシュを実行時に照合します。
- 検証済み Python 教材の入力データセットは `geocpuresearch/geocpu-v43-python-bridge-verified-20261104` です。`assets_manifest.json` と教材、独立再実行の記録を照合します。
- `--swe-proof-dir` は元の BF16 実行で取得済みの SWE 200 問の `COMPLETE.json` と原回答・採点記録 7 ファイルを含むディレクトリです。これは旧技能の履歴証拠であり、FP16 学習前後の直接比較には使いません。
- `--fixed44-proof-dir` は同じ親を FP16 で評価した固定 44 問の `COMPLETE.json` と `raw_answers_44_profiles.jsonl` を含むディレクトリです。
- `--control-claim` と `--python-claim` は別々の一回限りの `prelaunch_contract_once.json` です。各 claim は GPU 1 枚、精度 `train_fp16_gradscaler_v1`、実行上限 1,800 秒、保存用予備 180 秒を宣言します。二つの claim は同じ公式空き時間と先行ジョブの状態を参照します。公式に確認した空き GPU 時間は両 claim の予約時間の合計以上が必要です。
- 各 claim の `root_hold_ledger` には `paired_kernel_id`、二つの claim ID を辞書順に並べた `paired_claim_ids`、両予約時間の合計 `paired_reserved_device_seconds` を記録します。先行した BF16 SWE200 と FP16 固定 44 問の 2 ジョブは `active_jobs_snapshot.json` の `jobs` に `kernel_id`（または `ref` / `id`）と `status: COMPLETE` を持つ必要があります。v1 の stage 1 ジョブには `status: ERROR` が必要です。Kaggle のノートブック一覧の全ページを確認した記録として `all_pages_read: true` を付け、`confirmed_absent_kernel_ids` に保存失敗した v2 の完全な ID を入れる必要があります。v2 の ID が `jobs` にあれば起動用パッケージの作成を拒否します。ほかに稼働中のジョブがあれば、既存の起動前検証が拒否します。
- `--bridge-manifest` は新しい教材データセットの `assets_manifest.json` です。実際にマウントされたファイルは Kaggle 内でも再照合します。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python prepare.py `
  --swe-proof-dir <BF16_SWE200_OUTPUT>/actual `
  --fixed44-proof-dir <FP16_FIXED44_OUTPUT>/actual `
  --control-claim <CONTROL_CLAIM>/prelaunch_contract_once.json `
  --python-claim <PYTHON_CLAIM>/prelaunch_contract_once.json `
  --bridge-manifest <BRIDGE_DATASET_STAGE>/assets_manifest.json `
  --staged <NEW_ONE_USE_STAGE_DIRECTORY>
```

生成される `bootstrap.py` と `kernel-metadata.json` は、私有 Kaggle の T4 GPU 2 枚で実行するためのものです。`package-manifest.json` は使用したソース、証跡、claim、実行入口のハッシュを記録します。起動前に外部の資源台帳で二つの claim を消費済みに変更し、同じ claim で再提出しないでください。Kaggle 内に保存した claim は、提出時の記録を検証するための不変コピーです。

起動すると `CUDA_VISIBLE_DEVICES=0` に 0% 対照群、`CUDA_VISIBLE_DEVICES=1` に 1% Python 群を割り当てます。各群の `train_hybrid_stage_v2.py` は 250 回の更新、または宣言された時間上限で停止してチェックポイントを保存します。両群が 250 回を完了した場合だけ、`pair_status.json` は `BLOCK1_NEEDS_PAIRED_EVALUATION` を記録します。途中で止まった場合は各群のログと部分チェックポイントを確認してください。

## 検査

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest -v test_package.py
```

この検査は同梱するソースのハッシュ、実行入口の構文と GPU の割当て、両群の 4,096 位置の上限、Kaggle の名前とタイトルの 50 文字以内の制限、親評価が欠けた場合の拒否、二群の予約条件を確認します。実 GPU の学習結果を証明する検査ではありません。
