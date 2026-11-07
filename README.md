# GeoCPU diagnostics and research

GeoCPU／MORTRAの実験コード、診断、修正差分、検証原票を管理する非公開研究リポジトリです。

## 研究単位

- [v47：Python・知識の多段教材、基礎能力の保持、Colabライブ評価](research/v47_practical_20261107/README_JA.md)
- [v44fix：過去のPython導入診断](reports/v44fix/2026-11-03-python-cot.md)

v44の結果をv47の現在の性能と混同しません。各変更は別commit・PRで確認し、実際に測定した結果と仮説を分けます。

## 記録するもの

コード、CPUテスト、教材生成手順、実行ノートブック、データ/チェックポイントのハッシュ、原票の読み方、未完了事項を残します。大きな教材・評価問題は固定ZIP、重み・Adam状態はDriveの保存物を参照します。

新しい点数を報告する際は、code commit、checkpoint、問題、採点、生成予算を対応付けます。既存問題の変更や計測エラーを得点改善として扱いません。GitHub ActionsのCPU合格は、GPU学習やモデルの性能向上を意味しません。
