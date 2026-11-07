# 先にcolab_setup.pyの準備セルを実行してください。評価だけを開始します。
from pathlib import Path
import sys
import uuid
import torch
from IPython.display import display, Markdown
from live_logging import run_with_logging
from train_candidate import refuse_active_work
from compare_results import write_reports

if not torch.cuda.is_available():
    raise RuntimeError("ColabのランタイムをGPUに設定してから評価を実行してください")

WORK, STUDY = Path(WORK), Path(STUDY)
BASE_RUN = "v47_from28000_to30500_curriculum_v1"
NEW_RUN = "practical_20261107_10k_multistep_40500"
BASE_RESULTS = STUDY / "practical_20261107_10k_baseline_full_knowledge"
NEW_RESULTS = STUDY / f"{NEW_RUN}_eval_full_knowledge"
ORIGINAL_BASELINE = (
    STUDY / "v47_train_eval_suite_step_30500" / "original_tiers"
)


def evaluate(run, step, output, reuse_original=None):
    refuse_active_work(STUDY)
    locks = []
    for folder in (BASE_RESULTS, NEW_RESULTS):
        locks += list(folder.rglob("logger_active.json"))
        locks += list(folder.rglob("tier_stage_active.json"))
    if locks:
        raise RuntimeError(f"評価の稼働ロックがあります: {locks}")
    command = [
        sys.executable, "-u", str(WORK / "eval_candidate.py"), "run",
        "--study", str(STUDY), "--run", run, "--step", str(step),
        "--output", str(output), "--full-knowledge",
    ]
    if (output / "plan.json").is_file():
        command.append("--resume")
    elif output.exists():
        raise RuntimeError(f"評価計画のない既存フォルダーです: {output}")
    if reuse_original is not None:
        command += ["--reuse-original-from", str(reuse_original)]
    log_dir = (
        STUDY / "practical_20261107_10k_console"
        / f"eval_{step}_{uuid.uuid4().hex[:8]}"
    )
    rc = run_with_logging(command, log_dir, watch_root=output, cwd=WORK)
    if rc:
        raise RuntimeError(f"評価が停止しました。ログ: {log_dir}")
    if not (output / "original_tiers" / "complete.json").is_file():
        raise RuntimeError(f"元T1〜T8が未完了です。ログを確認: {output}")


evaluate(NEW_RUN, 40500, NEW_RESULTS)
evaluate(BASE_RUN, 30500, BASE_RESULTS, ORIGINAL_BASELINE)

REPORT_DIR = STUDY / f"{NEW_RUN}_comparison_{uuid.uuid4().hex[:8]}"
write_reports(BASE_RESULTS, NEW_RESULTS, REPORT_DIR)
display(Markdown((REPORT_DIR / "comparison.md").read_text()))
print("比較CSV:", REPORT_DIR / "comparison.csv")
print("比較JSON:", REPORT_DIR / "comparison.json")
