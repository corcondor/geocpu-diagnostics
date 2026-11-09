"""Builds colab_acquire_check.ipynb: the Google Colab notebook that acquires and checks the six approved Hugging Face datasets.

    python make_notebook.py [OUT_DIR]          # default: the folder of this file

The notebook carries its own code: acquire.py, checks.py, convert.py and pipeline.py are embedded as strings (FILES) and written to
/content/v49 by cell 3b, so Colab needs nothing but this one file (it is opened straight from GitHub; the notebook itself talks only to
huggingface.co and PyPI).  The four v48 'unseen_eval' sets (the model's own evaluation texts, 4 x 128 rows) are embedded too (EVAL_EXTRA,
only id / prompt / target / options / python_code of each row) because they are not stand-alone files on Drive and must be part of the
decontamination.  test_make_notebook.py compares every embedded text with its source.

CPU runtime only: the notebook metadata has no accelerator entry, no cell uses a GPU, and no secret is needed or read.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT.mkdir(parents=True, exist_ok=True)
NOTEBOOK = "colab_acquire_check.ipynb"
MODULES = ("acquire.py", "checks.py", "convert.py", "pipeline.py")
V48_DATA = HERE.parent / "v48_balanced_augmentation_20261108" / "data"
V48_FAMILIES = ("evidence", "japanese", "logic", "python")
V48_KEYS = ("id", "prompt", "target", "options", "python_code")

FILES = {n: (HERE / n).read_text(encoding="utf-8") for n in MODULES}


def v48_unseen_eval() -> dict:
    out = {}
    for fam in V48_FAMILIES:
        src = V48_DATA / f"{fam}_unseen_eval.jsonl"
        if not src.is_file():
            raise SystemExit(f"missing {src}: the v48 unseen_eval sets must be embedded for the decontamination")
        lines = []
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                lines.append(json.dumps({k: row[k] for k in V48_KEYS if k in row}, ensure_ascii=False))
        out[f"{fam}_unseen_eval.jsonl"] = "\n".join(lines) + "\n"
    return out


EVAL_EXTRA = v48_unseen_eval()


def md(src: str, cid: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str, cid: str, form: bool = False) -> dict:
    meta = {"cellView": "form", "jupyter": {"source_hidden": True}} if form else {}
    return {"cell_type": "code", "execution_count": None, "id": cid, "metadata": meta, "outputs": [], "source": src.splitlines(keepends=True)}


INTRO = """# GeoCPU v49: 外部データ6種の取得と検査（CPU だけ・GPU は使いません）

Hugging Face にある公開データ6種を Google Drive に取ってきて、検査して、GeoCPU の学習用の形（JSONL）に直します。
パソコンには何も取ってきません。重い処理はすべてこの Colab の上で動きます。

## 動くこと（上から順に）
1. **取得**（約1.9GB）: 6つのデータセットを、ライセンスを先に確認してから、コミットを固定して取得します。大きさと sha256 をハブの値と照合します。
2. **検査**: 件数をカードの値と照合し（違えば `MISMATCH` と大きく表示）、モデルの語彙に無い文字、長さ、読んで確かめるための100行を調べます。
3. **変換**: 1,900文字以内の GeoCPU の行に直します。コードは安全な枠の中で実行し、保存されていた出力を実行結果で置き換えます。
4. **絞り込み**: 完全一致と近い重複（MinHash）、評価データ（JMMLU・JamC-QA・T1-T8・v48 の評価セット・JaQuAD の dev・HotpotQA の validation）との13文字一致、旧コーパス423,348行との重複を取り除きます。
5. **記録**: `registry.json` と、日本語の表 `REPORT_JA.md` を書きます。採用する件数は、検査のあとで決まります（事前には決めていません）。

## 守っていること
- GPU は使いません。ランタイムは「ハードウェア アクセラレータ: なし（CPU）」のままにしてください。
- Drive には **新しいフォルダにだけ** 書きます（`external_data_v49/`、`external_data_v49_rows/`、`v49_acquire_20261109/`）。既にあるファイルを上書きも削除もしません。内容の違うファイルが既にあれば、そこで止まります。
- パスワード・API キー・トークンは要りません。通信先は huggingface.co（とその配信元）と PyPI だけです。
- 各セルの最後に `PASS` か `FAIL` が出ます。`FAIL` のときはそこで止まり、先へは進みません。
- 途中で止まっても、もう一度「すべてのセルを実行」で続きから動きます（取得済みのファイルは検証して飛ばし、変換済みの行は Drive から戻します）。

## 使い方
1. 「ランタイム → すべてのセルを実行」。セル3a で Drive のアクセス許可を聞かれます。
2. 7a（試運転）が終わると、全体の所要時間の目安が出ます（実測値ではなく見積もりです。コードの実行検証が一番長くなります）。
3. 最後に `REPORT_JA.md` の中身がセル10に出ます。読み方は README_JA.md にあります。

## 結果が違うフォルダに欲しいとき
内容が違うファイルが既にあって止まったときは、セル3a の `RUN_SUFFIX` を `'_2'` のように変えて、もう一度実行してください。新しいフォルダに書きます。
"""

CELL_PIP = '''# 2. 必要なライブラリ（PyPI から。GPU は使いません）
import importlib.metadata as md
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "pyarrow"], capture_output=True, text=True)
print((r.stdout + r.stderr)[-1500:])
assert r.returncode == 0, "FAIL [2 pip] pip install が失敗しました。上の出力を見てください"
print("PASS [2 pip] huggingface_hub", md.version("huggingface_hub"), "pyarrow", md.version("pyarrow"))
'''

CELL_CONFIG = '''# 3a. Google Drive をつなぎ、設定を決める（Colab 専用）
import os

try:
    from google.colab import drive
except ImportError as e:
    raise AssertionError("FAIL [3a drive] これは Google Colab の上で動かすノートブックです（パソコンでは動かしません）") from e
drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106"
assert os.path.isdir(DRIVE_ROOT), f"FAIL [3a drive] {DRIVE_ROOT} が見つかりません。Drive のフォルダ名を確かめてください"
RUN_SUFFIX = ""                 # 内容の違うファイルで止まったときだけ '_2' のように変える（新しい出力フォルダになります）
STOP_ON_COUNT_MISMATCH = False  # True にすると、件数がカードと違った時点で止まります（False: 大きく表示して記録し、続けます）
ONLY = None                     # 試運転で一部だけ動かすとき: ("jaquad", "toolace") など。None なら6つすべて
print("PASS [3a drive]", DRIVE_ROOT, "| RUN_SUFFIX =", repr(RUN_SUFFIX), "| ONLY =", ONLY)
'''

CELL_MODULES = '''#@title 3b. コードを書き出して読み込む（折りたたみ。中身は acquire.py / checks.py / convert.py / pipeline.py と同じです）
import sys
from pathlib import Path

FILES = __FILES__
EVAL_EXTRA = __EVAL_EXTRA__

CODE = Path("/content/v49")
CODE.mkdir(parents=True, exist_ok=True)
for name, text in FILES.items():
    (CODE / name).write_text(text, encoding="utf-8")
EXTRA = CODE / "eval_extra"
EXTRA.mkdir(exist_ok=True)
for name, text in EVAL_EXTRA.items():
    (EXTRA / name).write_text(text, encoding="utf-8")
sys.path.insert(0, str(CODE))
import acquire, checks, convert  # noqa: E401,F401
import pipeline as P

CFG = P.make_cfg(root=DRIVE_ROOT, suffix=RUN_SUFFIX, only=ONLY, stop_on_count_mismatch=STOP_ON_COUNT_MISMATCH, eval_extra_dir=str(EXTRA))
R3 = P.step_modules(CFG)
assert R3["pass"], R3["detail"]
'''

CELL_DRY = '''# 4. 試運転: 何をどれだけ取ってくるかだけを表示します（通信なし・書き込みなし）
R4 = P.step_dry_run(CFG)
assert R4["pass"], R4["detail"]
print("取得予定の合計バイト数:", f"{R4['expected_total_bytes']:,}")
'''

CELL_ACQUIRE = '''# 5. 6つを取得（再開できます）。ライセンス確認 → コミット固定 → 大きさと sha256 の照合 → Drive の新しいフォルダへ
R5 = P.step_acquire(CFG)
assert R5["pass"], R5["detail"]
'''

CELL_MIRROR = '''# 6a. 学習に使ってよいファイル（train）だけを、この Colab のディスクに写して sha256 を照合します（dev / validation は写しません）
R6a = P.step_mirror(CFG)
assert R6a["pass"], R6a["detail"]
'''

CELL_VOCAB = '''# 6b. モデルの語彙（3,898 件）と評価データの場所を確かめ、13文字の断片の索引を作ります
R6b = P.step_vocab_eval(CFG)
assert R6b["pass"], R6b["detail"]
'''

CELL_RAW = '''# 6c. 元データの検査: 件数をカードと照合（違えば MISMATCH）、語彙外の文字、長さ、読んで確かめる100行
R6c = P.step_raw_checks(CFG)
assert R6c["pass"], R6c["detail"]
for m in R6c["mismatches"]:
    print("MISMATCH:", m)
print("読んで確かめる100行（元データ）:", CFG.out / "reading")
'''

CELL_SMOKE = '''# 7a. 変換の試運転: 各データセットの最初の2,000件だけ。行の形・長さ・語彙・マスクを検査し、通らなければ止まります
R7a = P.step_convert_smoke(CFG)
assert R7a["pass"], R7a["detail"]
'''

CELL_FULL = '''# 7b. 変換の本番。絞り込み前の行は Drive の v49_acquire_20261109/converted_unfiltered/ に残します（再実行では Drive から戻します）
R7b = P.step_convert_full(CFG)
assert R7b["pass"], R7b["detail"]
'''

CELL_FILTERS = '''# 8. 絞り込み: 完全一致 → 近い重複（MinHash）→ 評価データとの13文字一致 → 旧コーパス423,348行との重複
R8 = P.step_filters(CFG)
assert R8["pass"], R8["detail"]
'''

CELL_FINAL = '''# 9. 最終の行（train / dev / heldout）を external_data_v49_rows/<データセット>/ に、registry.json と REPORT_JA.md を v49_acquire_20261109/ に書きます
R9 = P.step_finalize(CFG)
assert R9["pass"], R9["detail"]
'''

CELL_FINISH = '''# 10. 報告を表示し、Drive への書き込みを確実に終わらせます
rep = CFG.out / "REPORT_JA.md"
assert rep.is_file(), "FAIL [10 finish] REPORT_JA.md がありません。セル9を先に実行してください"
print(rep.read_text(encoding="utf-8")[:8000])
print("PASS [10 finish] 報告:", rep)
from google.colab import drive

drive.flush_and_unmount()
print("Drive の書き込みを終えてアンマウントしました。続けて作業するときは、セル3a から実行し直してください。")
'''


def py_literal(d: dict) -> str:
    """A dict of {name: text} as Python source with ONE SHORT LINE per source line (adjacent string literals), so that the notebook
    stays readable in diffs and the Colab editor never meets a 100 KB line.  ast.literal_eval gives back exactly the same texts."""
    out = ["{"]
    for name, text in d.items():
        out.append(f" {json.dumps(name)}: (")
        out.extend("  " + json.dumps(part, ensure_ascii=False) for part in (text.splitlines(keepends=True) or [""]))
        out.append(" ),")
    out.append("}")
    return chr(10).join(out)


def build() -> dict:
    files_literal = py_literal(FILES)
    extra_literal = py_literal(EVAL_EXTRA)
    head, rest = CELL_MODULES.split("__FILES__")                      # one pass: the literals themselves are never searched
    mid, tail = rest.split("__EVAL_EXTRA__")
    cell_modules = head + files_literal + mid + extra_literal + tail
    cells = [md(INTRO, "intro"), code(CELL_PIP, "pip"), code(CELL_CONFIG, "config"), code(cell_modules, "modules", form=True),
             md("## 4. 試運転\n取ってくるファイルと合計バイト数を表示するだけです。通信も書き込みもありません。\n", "md-dry"), code(CELL_DRY, "dryrun"),
             md("## 5. 取得\nライセンスをすべて確認してからデータを取ります。取得済みで正しいファイルは飛ばします。別の内容のファイルが既にあると止まります。\n", "md-acquire"),
             code(CELL_ACQUIRE, "acquire"),
             md("## 6. 元データの検査\n6a で train のファイルだけをこの Colab のディスクに写し、6b で語彙と評価データを確かめ、6c で件数・語彙外文字・長さを調べます。\n", "md-raw"),
             code(CELL_MIRROR, "mirror"), code(CELL_VOCAB, "vocab"), code(CELL_RAW, "raw"),
             md("## 7. 変換\n7a は最初の2,000件だけの試運転、7b が本番です。7b は時間がかかります（コードの実行検証）。\n", "md-convert"),
             code(CELL_SMOKE, "smoke"), code(CELL_FULL, "full"),
             md("## 8. 絞り込み\n重複と評価データとの重複を取り除きます。各段階の件数が表になります。\n", "md-filters"), code(CELL_FILTERS, "filters"),
             md("## 9. 書き出し\n最終の行・registry.json・REPORT_JA.md を Drive の新しいフォルダに書きます。\n", "md-final"), code(CELL_FINAL, "final"),
             md("## 10. 終わり\n", "md-finish"), code(CELL_FINISH, "finish")]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": NOTEBOOK, "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "cells": cells}


if __name__ == "__main__":
    nb = build()
    with open(OUT / NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:                 # LF on every platform (Windows would write CRLF)
        f.write(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("written", OUT / NOTEBOOK, (OUT / NOTEBOOK).stat().st_size, "bytes")
