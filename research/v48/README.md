# GeoCPU v48 — curriculum & live evaluation (research draft, 2026-11-07)

**目的:** v47の基礎能力を保存したままPython実行、資料読解と多段Binding、一般教養の知識利用を強化し、T1〜T8の全評価を**同じ問題・設定・採点器**で追跡する。元チェックポイント・稼働中Colab・Drive原本は書き換えない。

## 状況（検証と未検証の区別）

- **CPU検証済み:** `build_curriculum.py` から候補960件を決定的に生成。学習用640件 = Python実行512件 + 仮想業務資料の3段照合128件。dev/test各160件を分離。Pythonコードを独立実行して正解を検算し、全960件の`loss_mask_spans`をv47付属`prepare_30500.py`の`check_spans`で通過確認。
- **CPU検証済み:** `colab_live.py` の子プロセス実行、進捗行の抽出、`events.jsonl`・`status.json`保存。`compare_scorecards.py`は比較条件が違う場合に未比較と判定。
- **未検証:** v47の実`vocab`と全公式評価問題の排除照合、GPU学習への安全な組み込み、Colab上の長時間運転、Python/知識/検索の性能改善、JMMLU/JamC-QAスコアの改善。**このドラフトはそのまま本番学習へ投入しない。**
- **重要:** 候補データの「仮想業務資料」は実世界の事実知識ではなく、証拠を読んで結び付ける技能を学ぶもの。一般教養を強くするには、別途出典とライセンスを確認した実資料を取り込む必要がある。

## 1. 候補教材の生成（CPUのみ）

```bash
python build_curriculum.py --out /tmp/geocpu_v48_curriculum --seed 4807
python -m unittest discover -s tests -v
```

5群（CSV売上集計、JSON結合請求、コード修正、時間窓ログ集計、仮想資料の3段照合）に各 train 128 / dev 32 / test 32。学習セットに**既存のv47親データを上書き・混入しない**。親データのSHA、held-outの問題IDとテンプレート、v47 tokenizer正規化・2048長を検査し、独立パスへ安全な候補を作る`prepare_30500.py`に接続する前に、改めて検収が必要。

**実用的な知識・多段検索を伸ばす次の教材:** 出典つき段落を与えて「回答→追加検索→Python分析→根拠の要約」を選ぶ本物のワークフロー記録。`<SEARCH>`呼出し、検索結果、Binding、次の検索を別ステップとして記録し、検索結果・Python stdoutは**モデル生成させず損失マスク**する。未知の実資料は文書単位でtrain/dev/test分割。既存のTier 6に含まれる問題・近似テンプレートを訓練に含めない。

## 2. Colabで進捗をリアルタイム表示・永続記録

すでに稼働中の評価に**読み取り専用で取り付ける**場合：

```bash
python -u colab_live.py \
  --out /content/drive/MyDrive/GeoCPU_v48_logs/monitor_001 \
  --follow /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106/v47_train_eval_suite_step_30500/suite_【実ファイル名】.log
```

別の新規学習・評価を**明示的に開始するときだけ**：

```bash
python -u colab_live.py --out /content/drive/MyDrive/GeoCPU_v48_logs/trial_001 \
  -- python -u /content/【隔離した新規ランチャー】.py
```

- Colabセルには元のログ行が逐次出る。
- Driveに`console.log`、`events.jsonl`、原子的に更新する`status.json`が保存される。終了コード・エラー・T別スコア・Python実行状態を区別。
- `status.json`の途中経過は正式な最終得点ではない。重複行の集計や後出し解答を避けるため、最終スコアは各公式採点JSONからのみ確定。
- 出力先には**毎回空のディレクトリ**を使う（誤って過去ログを上書きしない）。ファイルの更新は10秒間隔を目安にする。Colabセルを止めると監視も停止する。

## 3. T1〜T8の同一評価で比較

- **T1** SWE数値一致・終了ステップ、**T2** 状態追跡、**T3** JamC-QA、**T4** JMMLU/ソフトウェア状態計算、**T5** 論理演算、**T6** 検索最終正答・検索経路、**T7** closed/open-book知識、**T8** Python/TIR。
- **v47原本評価の参考値**は`data/v47_30500_reference.json`に記録した。ただしT8は異なるモードの結果を含むため安易な同系列比較不可。
- 旧T1〜T8の問題・評価器は変更せず、新規拡張問題（別コホート）を追加する。同じ問題が繰り返される旧Tier 6は、旧指標として存続し、独立した新規問題を**新しい指標名**で集計。
- `compare_scorecards.py` は**コホートSHA・採点器ID・デコード設定ID・問題数が完全一致するときだけ**前後差を計算する。条件違いは`not_comparable`。未測定項目を0%として扱わない。

```bash
python compare_scorecards.py --base frozen_v47.json \
  --candidate frozen_v48.json --out /tmp/comparison_v48.json
```

## 4. 比較計画（未実施）

1. v47 Step 30,500重みを読み取り専用で固定し、T1〜T8全スイートを一度だけ公式ハッシュとともに記録する。
2. 既存教材のみの継続学習Aと、既存教材 + 新しい多段教材Bを**同じ初期チェックポイント、同じtoken/FLOP予算・seed**で別枝学習する。v47旧データの比率を確保し、旧能力を保持する。
3. 同じT1〜T8と新規の独立資料課題で前後比較する。知識評価は検索なし/自律検索/検索強制/正解文書を渡すOracleの4条件を別指標として記録する。
4. **暫定の失敗基準:** SWE数値一致や状態追跡が2ポイントを超えて悪化、検索実行回数だけ増えて最終正答不変、Pythonのコード実行成功と最終数値正答が改善しない場合はデータ混合案の採用を見送る。0.5〜2ポイントの差は標本誤差と採点の再現性を確認。

## 5. 研究上の安全境界

- GitHubはレビュー用の**private `geocpu-diagnostics`ドラフトPR**。公開の`mortra`へ自動転記しない。
- コード、種、生成マニフェストだけをGitHubに置き、巨大なチェックポイント・実験者のAPIキー・Colab認証・非公開の学習データはコミットしない。
- 今回はGPU学習・モデル推論・本番Colab監視を実施していない。性能の向上は一切主張しない。