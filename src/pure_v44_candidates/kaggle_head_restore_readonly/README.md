# 純 v44 の出力ヘッド一時復元：Kaggle 隔離移植

この候補は、純 Python v44 Hotfix の親 step 15,000、子 step 25,000、子の上流計算に親の `base.head` だけを一時的に使う三群を、同じ生成コード・同じ64問で比較します。元のチェックポイントと学習データには書き込みません。既存のハイブリッド v44fix Kaggle 二件の回答は再利用しません。

`package_sources.py` は小さいソース ZIP と、そのZIPのSHAを固定した**起動不可の下書き入口** `kaggle_pure_head_restore_entrypoint_v1.py` を作ります。下書きは公式GPU残量の確認票が無いので実行を拒否します。チェックポイントを同梱しません。入力資産は読み取り専用の Kaggle Dataset へ別途配置する必要があります。Dataset のルート直下には、`source_manifest.json` に固定した `GeoCPU_Frontier_CoTraining_v43` と `GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103` の相対パスを保ってください。親・子チェックポイントの合計は約 3.96 GB です。実行時には 8 GB 以上の空き作業領域と CUDA GPU を確認します。

実際の送信準備には、ソースZIPと入力資産を別々のKaggle Datasetに配置してから `prelaunch_gate.py` を実行します。`--hold-ledger` には、1時間以内に再照合したRootの未返却GPU時間台帳を指定します。台帳は `geocpu_kaggle_unreturned_holds_v1` 形式で、`reconciled_utc`、証拠ファイルのパスとSHAを含む `holds`、別ファイルの照合記録を示す `reconciliation_evidence_path` と `reconciliation_evidence_sha256` を持ちます。照合記録には `geocpu_kaggle_hold_reconciliation_v1` 形式、`checked_utc`、台帳と同じ `unreturned_device_seconds` が必要です。未返却が0の場合も、照合記録に `zero_hold_evidence: true` が必要です。台帳や証拠の欠落を0時間と扱いません。

入口はKaggle公式のGPU残量を**その場で再取得**し、既存のv43・ハイブリッドv44fix二件のジョブがともに終了していることを確認します。無料分の計算には公式の `minimum_time_allowed` だけを使います。公式の `time_used`、`time_reserved`、Root台帳の未返却時間、安全余裕を引きます。`total_time_allowed` に含まれうる有料拡張分は無料残量へ加えません。公式APIが失敗した場合、どちらかが実行中の場合、残量が足りない場合は、実行可能な入口とKaggleメタデータを作りません。通過した場合だけ、新しい送信ディレクトリへ確認票を埋め込んだ入口と `kernel-metadata.json` を作ります。確認票が1時間を超えて古くなれば、Kaggle側の入口も実行を拒否します。入口はジョブを送信しません。Kaggleへの送信は別の明示操作です。

GPU残量は2台分の使用時間として保守的に計算し、安全余裕を差し引きます。今回の上限は8時間ですが、公式残量が少なければ今回の実行時間だけを短くします。64問×3群と各回答2,048トークンの評価範囲は縮めません。外側の `timeout` は選んだ実行時間で停止し、内側はその5分前に止まります。内側の期限では `PARTIAL.json` を残して正常終了し、Kaggleに部分回答を保存させます。各回答の期限は15分です。CPUで行う静的事前検査は、ソースZIPを展開して `python runner.py --preflight --assets-root <同じ構造の資産ルート> --output-dir <未使用パス>` で実行できます。

出力は開始時から `PARTIAL.json` で未完了と示します。各回答と三群組を原子的に保存します。次の明示起動では、前回のKaggle出力を `--resume-kernel-source` で入力へ添付します。入口は前回の `PARTIAL.json` がちょうど1件ある場合だけ作業領域へ複製します。固定した資産パス・SHA・64問の全メタデータ・生成設定を照合して、欠けた回答だけを生成します。既存の回答は再生成も上書きもしません。全64組とハッシュを照合して `COMPLETE.json` を最後に作成します。完全性は `python verify_outputs.py` の `verify_complete` で再検査できます。採点の正否と Python の実行成功率は、この診断では測りません。

三群の比較からは「子の上流状態のまま親の出力ヘッドを使うと改善するか」を測れます。親の上流状態に子のヘッドを使う逆方向はこの候補に含まれないため、ヘッドと上流の相互作用を完全には分離できません。
