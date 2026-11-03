# GeoCPU v44fix — Colab で回す手順（A100 推奨）

読むだけの入力：`MyDrive/GeoCPU_Frontier_CoTraining_v43/`（v43 の zip と `checkpoint_step_15000.pt`）。
書き出し先：`MyDrive/GeoCPU_v44fix_20261103/run_1/` だけ。v43・v44 の元ファイルには一切書き込まない。

## セル1：ゲート（まずこれだけ）
公開データ準備 → v43 形式への変換と混合 → v43 の基準評価 → 1000 ステップ試運転（15000→16000）→ 合否判定。
途中で切れても、もう一度実行すれば終わった段階は飛ばして続きから再開する。

```python
import os, sys, zipfile
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive; drive.mount('/content/drive')
ZIP = '/content/drive/MyDrive/GeoCPU_v44fix_20261103/GeoCPU_v44fix_bundle.zip'
W = '/content/v44fix'
os.makedirs(W, exist_ok=True)
zipfile.ZipFile(ZIP).extractall(W)          # コードは毎回上書き、data/ は残る
os.chdir(W); sys.path.insert(0, W)
import importlib, colab_v44fix; importlib.reload(colab_v44fix)
result = colab_v44fix.main(full=False)
```

最後に `GATE PASSED` か `GATE FAILED` が出る。合否の根拠は `run_1/gate_16000.json`、基準（v43）は `run_1/gate_baseline_v43.json`。

合格条件（v43 比）：
- SWE の正解率の低下が 5pt 以内
- 状態追跡の正解率の低下が 5pt 以内
- ALU 計算の正解の減少が8問中1問以内
- 反復率の上昇が 10pt 以内

TIR（Python を使う数学）は記録するだけで、ゲートの条件には入れていない（1000 ステップではまだ育たないため）。

## セル2：本番（セル1が PASS のときだけ）
v43 の step15000 から 25000 まで、1万ステップを途切れずに学習する。終わったら v43 の 7 階層ベンチと最終判定まで自動で回す。

```python
result = colab_v44fix.main(full=True)
```

## 失敗したとき
`GATE FAILED` なら本番は自動で始まらない。`run_1/gate_16000.json` の `checks` で、どの条件で落ちたかを確認する。
