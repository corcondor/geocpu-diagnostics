# 基礎能力を維持しながらPython・検索利用を増やす試行

## このパッケージで実装したもの

保存済みv47 step 30,500から、**新しい枝だけ**で追加学習するランチャーです。原本の422,500行はバイト単位で保ち、新しいPython教材と知識・多段検索教材を追加します。採用行数はそれぞれのmanifestと実ファイルで照合し、学習入力manifestにも記録します。元のT1〜T8、Python120、工学・可視化、検索診断の設問は教材へ取り込みません。

旧データのSHA-256は `358f96521122606ce398eb6f5f30f51e71048728e52e6b8ee9f55de854fcd5fc`、親モデルは `cc38411e66aab0ed309d0fa6e0b268d04317da17ffa47f730383e6ad0e2d0391` と照合します。親のAdam状態とPython/NumPy/PyTorch/CUDAの乱数状態も復元します。教材が増えるためsamplerは新たに開始し、その点を記録します。

**実装とCPU検証を行った候補です。GPU学習・学習後のモデル得点は未測定です。** 基礎能力が保たれたかは、同じ問題と同じ条件での学習前後評価で判断します。

## 維持する機構と今回の変更

| 項目 | 設定 |
| --- | --- |
| CED・Delta記憶・Binding・内部反復・ALU・検索器 | 原本の構造を維持 |
| 学習対象 | 原本と同じ全学習パラメータ。固定ALU/RAMに学習パラメータが現れた場合は停止 |
| 学習率 | `3e-6`で固定 |
| 最初の試行 | 1,000更新（30,500→31,500） |
| 更新あたりの予算 | 元と同じ36,864トークン、batch上限64 |
| 文脈 | 元と同じ2,048位置 |
| 新教材 | 各manifestで固定された採用行。初期設定の提示倍率16、実際の提示回数を計測 |
| 結果観測 | Python `output`、検索 `RESULT`は入力として読み、損失から除外 |
| リプレイ | 旧データを全て保持。試行全体で旧教材の教師トークン比率75%以上を要求 |
| 教師モデル | 学習開始時の親step 30,500を凍結 |
| KL | 旧教材の全groupの教師トークンだけに適用。新教材には適用しない |
| OOM | 原本と同じ全ステップ再計算。途中行を捨てる処理を追加しない |

**KLの適用範囲は今回の変更です。** 元はv43の一部旧行にv43教師を当てていました。この候補は、今の親が獲得したPython・検索の挙動も維持するため、親30,500を教師として既存group全体に当てます。KLの式、optimizer、forward、勾配計算、OOM時の再試行は変えていません。

この保護設定だけで退行がなくなるとは言えません。T1/T2/T5や既存の検索・日本語出力を保てなければ、低い学習率のまま新教材の提示倍率やKL重みを検討します。元の親と旧試験結果はそのまま残します。

## 3つの比較条件

| variant | 教材 | 目的 |
| --- | --- | --- |
| `replay_only` | 旧422,500行のみ | 追加更新だけで変わる分を測る |
| `answer_only` | 旧行＋同じ新問題の最終回答だけ | 多段操作の教師信号の有無を測る |
| `multistep` | 旧行＋検証済みの操作・観測・最終回答 | 実用的な手順を学ぶ候補 |

3条件とも同じ親、1,000更新、トークン予算、低学習率、教師保護を使います。`answer_only`でも問題文は変えません。全行の文脈長とマスクが変わるため実際の教師トークン数・提示数は同一になりません。各条件の実測投入量をログに残し、差の原因を検討します。`replay_only`には新教材がないため「新教材1,000回提示」の条件は適用せず、それ以外の条件は維持します。

原本のsamplerで試行全体を事前に再現し、以下のどちらかを満たさなければ**最初の更新前に停止**します。

- `multistep`/`answer_only`の新教材提示が1,000回以上。
- 旧教材の教師トークンが全教師トークンの75%以上。

行数比率だけから「75%維持」と判断しません。ツール観測を除外し、長文分割後の実際のsampler対象で比率を計算します。保持する行数と、今回の1,000更新で実際に読む行数は別です。

## Colabでの最小手順

Driveをマウントし、パッケージを `/content/geocpu_delivery` に展開してください。稼働中の学習・評価がある場合は、まず `live_logging.py watch` で進捗を確認します。ランチャーは他のプロセスやDrive上の稼働ロックを検出した場合、新しい学習を開始しません。元の処理を止めたり、ロックを自動削除したりしません。

### 1. CPU事前確認

```bash
cd /content/geocpu_delivery
python -u train_candidate.py \
  --study /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106 \
  --output /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106/practical_multistep_pilot_31500 \
  --variant multistep --steps 1000 --new-repeat 16 \
  --preflight-only \
  --preflight-report /content/practical_multistep_preflight.json
```

事前確認は元のデータを読み、ローカル作業フォルダーに新しいデータと差分を作ります。`--output`の新枝はまだ作成しないため、次の実行で同じ出力先を使えます。`--preflight-report`は既に存在するファイルを上書きしません。

### 2. 空いている元と同じA100で試行

```bash
python -u train_candidate.py \
  --study /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106 \
  --output /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106/practical_multistep_pilot_31500 \
  --variant multistep --steps 1000 --new-repeat 16
```

既存の再開条件を保つため、単一の `NVIDIA A100-SXM4-40GB` を確認します。別GPUへ黙って切り替えません。`--output`は新しいフォルダーであることが必要です。他条件は別の新規出力先で `--variant answer_only`、`--variant replay_only` を指定します。

キャッシュが残っていれば `--prepared-parent /tmp/geocpu_30500_.../train_30500.jsonl` を指定できます。内容のSHAが親データと完全一致する場合だけ使用します。キャッシュがない場合は、以前の正規の再構成コードを呼び出し、旧28,000データ＋既存372行から旧30,500データを再現して最終SHAを照合します。

### 3. 学習後の採点

学習の最後は、モデル・Adam・乱数/sampler状態を新枝に保存して `evaluation: pending` と記録します。その保存チェックポイントを、同じT1〜T8とPython120、別枠の追加評価に渡してください。ランチャーは、低い学習損失を「得点改善」や「基礎能力維持」と判定しません。

## 毎回必要な元ファイル

このパッケージは大きいモデル重みや旧全教材を二重に添付しません。以下を読みます。

- `GeoCPU_v47_fullmodel_20261106/` 配下の `GeoCPU_v47_fullmodel_bundle_v1_READY.zip`、`GeoCPU_v47_continue30500_train_eval_v1.zip`、`GeoCPU_v47_python_science_next_v1.zip`、`v47_continue25500_training_v1.zip`。
- 親 `v47_from28000_to30500_curriculum_v1/` の `input_manifest.json`、`training_saved.json`、`fork_from_28000.json`、step30,500のcheckpoint/optimizer/train_state/receipt。
- キャッシュを使わず再構成する場合、旧20,500→25,500→28,000枝の保存原票と、原本 `GeoCPU_Frontier_CoTraining_v43/Grand_cotraining_v43.zip` および `checkpoints/checkpoint_step_15000.pt`、`GeoCPU_v44fix_20261103/run_8_r5b/bridge/bridge_train.jsonl` と `bridge_heldout.jsonl`。これは元の教材再構成が必要とする依存物で、新しい教師は30,500です。
- パッケージ内の `frozen/`、`data/practical_*`、`data/knowledge_*`。原テスト、追加dev/eval、教材manifestを含みます。

親と元ZIPの照合値はコード内と原票で確認します。ファイル不足や不一致の場合、代わりのチェックポイントや短い教材へ自動退避しません。

## ログと計算費用

`training_events.jsonl`に10更新ごとに次を保存し、Colabの表示にも同じJSONを流します。

- 全体の更新数、損失、親教師とのKL、実際の学習率、経過時間とETA。
- 教材sourceごとの実際の提示回数、入力トークン数、教師トークン数。
- 旧教材の教師トークン比率、確保VRAM、ピークVRAM、OOM再計算回数。

`console/console.log`に標準出力・標準エラーを保存し、子が無言でも5秒ごとにheartbeatを表示します。終了時に新枝へ学習入力manifest、原本とのruntime差分、fork原票、チェックポイント、Adam、乱数状態、receiptを保存します。

1,000更新のsampler予算の上限は **36,864,000トークン枠**です。実際の非padding入力・教師トークンは別途記録します。壁時計時間はこの候補では未測定です。旧実行の速度をこの候補の実測値として使いません。

パラメータを追加しません。凍結教師を含む学習機構は既にありますが、KLを適用する旧教材の範囲を増やすため教師のforward量は増え得ます。247,568,664パラメータの教師重みはfloat32なら約0.92GiBです（計算値）。新しいteacherを既存teacherと置換するので教師重みの個数は増えません。バッチ長・対象行による一時VRAM、追加forward時間はColabで確認します。元と新枝を両方保存するディスク容量も必要です。

## 失敗時の扱い

事前の提示数不足、75%の旧教師トークン条件不足、語彙/マスク不整合、元ファイルのSHA不一致は更新前に停止します。OOMは原本の全ステップ再計算、非有限loss/勾配はその更新を拒否します。失敗時は新枝のログとローカル作業フォルダーを残し、元の重みを変更しません。

学習後の採用基準は固定評価で定めます。Pythonや知識の得点が上がっても、維持対象のT1/T2/T5などが合意した閾値を超えて低下した候補を自動採用しません。公式選択肢スコア、検索ありの実用スコア、Python実行の成功、最終回答正解は別々に比較します。

## 今回の検証範囲

CPUテストは旧行のbyte-exact保持、元テスト混入拒否、観測マスク、answer-only対照、samplerの事前計算、元のforward/KL/optimizer retryのAST一致、実行中プロセスの読み取り専用検知、比較条件の引数を確認します。

この作業環境にはPyTorchがなく、Driveの学習中ランタイムにも接続していません。**実際の親データ全422,500行の再構成と、その全量を使うsamplerのCPU previewも未実行です。** 親データ本体と旧Grand/bridge依存をこの環境で取得できていないため、Colab上の元依存一式またはSHAが一致する既存cacheからの検証が残っています。依存不足やSHA不一致は学習更新前に停止します。GPU学習、基礎能力の保持と得点向上は未測定です。

`knowledge_dev.jsonl`と自由回答の`knowledge_free_dev.jsonl`は両方を学習混入検査へ渡し、追加学習には使いません。
