# Colab準備セル。保存済みモデルから評価するためのコードだけを復元します。
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import hashlib
import importlib
import sys
import tempfile
import zipfile

STUDY = Path("/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106")
ARCHIVE = (
    STUDY / "GeoCPU_practical_training_logging_20261107"
    / "GeoCPU_practical_training_logging_20261107_10k.zip"
)
EXPECTED_SHA256 = "939e75d61aba5cd9f271b46e3d210d42af5a0c51904b80df101dbf40072d55d5"

if not ARCHIVE.is_file():
    raise FileNotFoundError(f"実行用ZIPが見つかりません: {ARCHIVE}")
if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("実行用ZIPのSHA-256が一致しません")

WORK = Path(tempfile.mkdtemp(prefix="geocpu_eval40500_", dir="/content"))
with zipfile.ZipFile(ARCHIVE) as archive:
    archive.extractall(WORK)

# 再実行時も、今回のWORKにある同梱モジュールを読み込みます。
for name in (
    "live_logging", "prepare_training", "frozen_eval",
    "train_candidate", "eval_candidate", "compare_results",
):
    sys.modules.pop(name, None)
sys.path.insert(0, str(WORK))
importlib.invalidate_caches()

from live_logging import run_with_logging
from train_candidate import refuse_active_work
from compare_results import write_reports

print("準備完了:", WORK)
print("続けて評価セルを実行してください。学習のやり直しは不要です。")
