# GeoCPU v47：実用教材・基礎能力の保持・ライブ評価

保存済み **step 30,500** を起点に、Pythonと検索利用を伸ばすための追加学習候補です。既存T1〜T8をそのまま比較し、Colabで処理件数と実際の正答数を見られるようにします。

**実装・教材のCPU検算は完了しています。新しいGPU学習、学習後の得点、基礎能力の保持は未測定です。** 検証原票は `verification_cpu.json`、未完了事項は `RESEARCH_LOG.md` に記載します。

## 最初に使うもの

| 目的 | ファイル |
| --- | --- |
| 今動いている評価の進捗を見る | `GeoCPU_live_monitor_20261107.ipynb` |
| 新枝で追加学習して全比較する | `GeoCPU_practical_training_logging_20261107.ipynb` |
| 今回の判断と限界を読む | `GeoCPU_practical_training_logging_brief_20261107.md` |

監視ノートブックは**別のCPU Colabセッション**で開きます。実行中のGPUセッションと同じカーネルでは、新しいセルは実行待ちになります。監視は元ファイルを読むだけです。

実行用ZIPはDriveのプロジェクト内 `GeoCPU_practical_training_logging_20261107/` に置く構成です。ノートブックの `ARCHIVE` で別の保存場所も指定できます。ZIPをハッシュ検証して新しい `/content` 作業フォルダーへ展開し、稼働中のソースへ重ねません。

[実行ファイルとノートブックの保存先](https://drive.google.com/drive/folders/1XAsTMUU6iSCwTl9-moU-ffms5IegBUfa)

## 実装した内容

- **教材**：Python512行（16種類の実務処理・修正課題）、知識336行（168連鎖の2回答形式）。実際の道具の結果を読み、次の操作と最終回答を出す教師列を作ります。観測文自体は予測損失から除外します。
- **保持**：旧422,500行を残す設計、親30,500を固定した教師、低学習率、実samplerで旧教師トークン75%以上を確認。構造や文脈長2,048は維持します。
- **比較**：原T1〜T8、Python120、工学64、可視化10、検索200/198、公開dev80、既存公開9群を登録。新しい実用評価240問は別の集計です。
- **知識の切り分け**：既存の閉域試験に加え、64連鎖を検索なし／検索あり／根拠を与えた状態で比較。全JMMLU/JamC-QAの検索あり生成比較も選択できます。
- **ログ**：一問ごとの原回答・採点・エラー、処理済み件数、正答数と分母、Python実行成功、最終回答正解、経過時間、ETA、5秒ごとの生存確認。

知識教材は既存の訓練用事実から検索手順を増やしたものです。一般教養全分野の新しい知識を網羅的に追加したという意味ではありません。Python教材の行数も独立した問題種類の数ではありません。詳細は `CURRICULUM.md` と `KNOWLEDGE.md` にあります。

## GitHubで研究を引き継ぐ

研究用リポジトリは非公開の `corcondor/geocpu-diagnostics` です。今回のコードは `research/v47_practical_20261107/` に置き、過去v44の修正と別のPRで比較します。

GitHubにはコード、テスト、ノートブック、入力ハッシュ、CPU検証結果、研究記録を置きます。大きな教材・評価問題本体は固定ZIPから読み込みます。重み・Adam状態は既存Drive上の保存物をハッシュ確認して使います。

リポジトリから使う場合、同じ版のZIPを指定して不足するデータを復元できます。

```bash
cd research/v47_practical_20261107
python hydrate_artifact.py --archive /path/to/GeoCPU_practical_training_logging_20261107.zip --check-only
python hydrate_artifact.py --archive /path/to/GeoCPU_practical_training_logging_20261107.zip
python -m unittest discover -s tests -v
```

復元は `data/` と `frozen/` の不足ファイルだけが対象です。既存ファイルがハッシュと違う場合は停止します。GitHub Actionsでは、データを必要としないログ・復元処理のCPUテストを実行する構成です。GPU性能テストとは別です。

## 学習を始める前に必要な確認

元モデル・元ZIPの存在に加えて、旧全教材の正確なキャッシュ、または原本Grand ZIPとbridge教材を含む再構成依存が必要です。この作業環境では旧422,500行全体を再構成できていないため、**実際の親データによるsampling事前確認はColabで未完了**です。`TRAINING.md` の依存物と停止条件を参照してください。

最初は1,000更新、同じ親から `multistep`／`answer_only`／`replay_only` を比較できる設計です。今回の保持判定は工学的な目安で、保持の保証ではありません。比較条件が違う、未計測がある、原票が壊れている場合は改善判定を出しません。

## ファイル案内

| 機能 | 実装 | 説明 |
| --- | --- | --- |
| Python教材 | `build_curriculum.py` | `CURRICULUM.md` |
| 多段検索教材 | `build_knowledge.py` | `KNOWLEDGE.md` |
| 学習準備・新枝 | `prepare_training.py`, `train_candidate.py`, `runtime_practical_v1.py` | `TRAINING.md` |
| 既存試験の固定 | `frozen_eval.py`, `frozen/registry.json` | `FROZEN_EVAL.md` |
| 評価と比較 | `eval_candidate.py`, `compare_results.py` | 本ノートブックと技術ブリーフ |
| ライブ監視 | `live_logging.py` | `LOGGER_INTEGRATION.md` |
| 入力復元 | `hydrate_artifact.py`, `PACKAGE_MANIFEST.json` | このREADME |

