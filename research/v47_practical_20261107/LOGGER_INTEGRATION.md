# 評価ログとColabの進捗表示

`live_logging.py` はCPUだけで動作し、モデル、チェックポイント、採点関数を変更しません。GPU学習・モデル評価の結果を作ったことにはなりません。

## 今動いているv47評価を、そのまま見る

現在の評価コードはPythonの各問を `python_120/partials_step_30500/*.json` に保存し、コンソール表示だけを10問ごとに行います。監視コードは保存済みの各問の採点を読み、5秒ごとに表示を更新できます。

別のColabセッションでDriveをマウントして、次を実行します。稼働中の評価を止めて同じセルを置き換える必要はありません。同じランタイムで評価セルが実行中の場合、新しいセルは評価終了まで実行待ちになるため、別セッションから見るか、次回から下のラッパーを使います。

```python
from live_logging import watch_existing

watch_existing(
    "/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106/v47_train_eval_suite_step_30500",
    poll_s=5,
    notebook=True,
)
```

停止ボタン／Ctrl-Cは監視だけを終了します。元の評価プロセスにシグナルを送りません。Google Driveの同期遅延があるため、別セッションからの表示はファイル到着後の更新です。

CLIで一度だけ確認する場合:

```bash
python -u live_logging.py watch /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106/v47_train_eval_suite_step_30500 --once
```

表示内容:

- 各チェックポイントの処理件数／予定件数。
- 最終数値の一致、EOS付き最終回答の一致、Python実行成功、ツール結果と最終回答の両方の一致。これらを異なる指標として表示。
- T1〜T7の既存ログに明示された途中の正答数と処理件数。単なる `90/120` を90問正解とは解釈しない。
- T8のコード形式・説明形式・Python呼出指標。T8に存在しない数値正答率は作らない。
- 工学・可視化の採点済み件数と、依存関係不足などの未測定件数。
- 元ログの最終更新からの経過秒数、読み取りエラー。

## 新しい評価の一問ごとのログ

```python
from live_logging import RunLogger, file_sha

identity = {
    "suite": "Python120-frozen",
    "mode": "python-enabled-search-disabled",
    "checkpoint_id": "candidate_step_31000",
    "checkpoint_sha256": file_sha(checkpoint_path),
    "dataset_sha256": file_sha(frozen_questions_path),
    "scorer_sha256": file_sha(original_scorer_path),
    "runtime_sha256": file_sha(inference_runtime_path),
    "budget": {"max_tokens": 2048, "enable_search": False,
               "enable_python": True, "loop_guard": False,
               "repetition_penalty": 1.0},
    "seed": 42,
    "metric": "terminal_explicit_exact",
}

with RunLogger(new_output_directory, identity, total=len(cases)) as log:
    for case in cases:
        if not log.start_case(case["id"]):
            continue
        generated = original_generate(case)
        scored = original_score(case, generated)
        # raw_refは先に保存・flushした実際の原回答ファイル。
        log.result(case["id"], correct=bool(scored["terminal_explicit_exact"]),
                   metrics={"python_execution_success": scored["python_execution_success"],
                            "tool_and_terminal_exact": scored["tool_and_terminal_exact"]},
                   raw_ref=str(saved_raw_answer))
```

実際のチェックポイント・データ・採点コード・推論コードのSHA-256が必要です。未知のSHAをダミー値で埋めないでください。コードは実装箇所の例です。`original_generate`、`original_score`、`saved_raw_answer` は評価器側の実処理を接続します。

新しい出力先に以下を保存します。

| ファイル | 内容 |
| --- | --- |
| `run.json` | チェックポイント、問題集合、採点、実行設定の固定情報 |
| `events.jsonl` | 開始、一問開始、一問の採点、heartbeat、終了を追記保存 |
| `status.json` | 各問とheartbeatで原子的に更新する現在値 |

通常は各問の結果を即時保存し、画面は最大1秒に1回更新します。一問の生成が長くても別スレッドが5秒ごとにheartbeatを保存します。heartbeatは生存確認であり、モデルが有意な進展をしたという主張ではありません。

### 件数の意味

- `processed`: 採点結果、無効回答、未測定、評価エラーを含む処理済み件数。
- `answered`: `scored + invalid`。モデルが形式に従わなかった無効回答を除外して正答率を上げることはしません。
- `correct / answered`: 既存採点規則が正解とした割合。
- `incorrect`: 形式が有効な誤答。`invalid`とは別に表示。
- `unmeasured`: CAD依存関係不足などで評価できなかったもの。
- `error`: 採点コード例外など。モデルの誤答とは扱いません。
- 未測定・エラーがある評価は、全問測定済みの比較表には受け入れません。正答率だけでなく `answered / total` の測定率も保存します。

元の採点が形式不正を単に `False` と返す場合は、そのまま `scored, correct=False` を渡します。既存スコアの意味を変えません。`invalid`は元の採点器が形式不正と明示して区別できる場合だけ使用します。

### 再開

同じ設定なら `resume=True` で再開できます。採点済みIDは二重集計せず、矛盾する再採点やデータ・採点コード・生成上限変更は拒否します。開始記録だけがある問は保留に戻し、誤答にはしません。

異常終了で `logger_active.json` が残った場合は自動削除しません。元のログ書込プロセスが終了したことを確認した後、その新規結果ディレクトリ内のロックだけを削除して再開するか、新しい出力先を使います。途中で切れたJSONL行も黙って削除しません。

## 既存評価を次回からバッファなしで実行

```python
from live_logging import run_with_logging
import sys

returncode = run_with_logging(
    [sys.executable, "/content/your_verified_evaluator.py", "--output", new_results],
    new_console_output_directory,
    watch_root=already_created_results_directory,
)
```

ラッパーは標準出力と標準エラーを即時表示・追記し、子プロセスが無言でも5秒ごとに更新します。`python -u`と`PYTHONUNBUFFERED=1`を使用します。ラッパーの出力先は、読み取り専用で監視する元の結果ツリーの外に置きます。`watch_root`がまだなければ、子プロセスが結果フォルダーを作るまで待ってから読み取りを開始します。ラッパー自身は監視先を作りません。ラッパーの停止は、このラッパー自身が明示的に起動した子プロセスだけを終了させます。

## 同じ条件の比較CSV

```bash
python -u live_logging.py compare /content/results/baseline /content/results/candidate --out /content/results/python_comparison.csv
```

問題集合、問題ID、採点コード、推論コード、モード、生成予算、seed、指標が一致し、全問測定が完了しているときだけ出力します。チェックポイント間の正答率差に加え、同一問題で改善した件数／退行した件数を保存します。T1とT2、検索なしと検索あり、元テストと追加テストはそれぞれ別の比較です。

得点は`events.jsonl`の各問題結果から再集計し、完了時の`status.json`と照合します。statusだけの得点変更、未完了、イベント欠損、同じ件数でも異なる問題IDは比較を通しません。Pythonの副指標は、その指標が実際に保存された問題数を分母にし、全問で観測されていなければ改善差分を出しません。

`compare_results.py`は元T1〜T8と補助評価を一つの表に並べます。T1/T4の同名ログ行はTier区間ごとに解析し、T3の丸めた割合から正答数を逆算しません。T4は元の全8,100行のraw結果、T8は二つの形式評価profileを別々に扱います。元評価を再利用する場合は、元のstageコードと元の出力pathに対応する完了markerを検証します。コピー先に合わせてmarkerを書き換えません。

## CPU検証

```bash
python -m unittest discover -s tests -p test_live_logging.py -v
```

検証対象は分母、途中停止と再開、二重集計、条件違いの比較拒否、静かな子プロセス中のheartbeat、既存結果の読み取り専用監視です。GPUモデルの得点検証とは別です。
