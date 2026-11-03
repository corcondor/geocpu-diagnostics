"""Minimal Kaggle CPU environment probe. No public source code is executed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requested_metadata(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Kernel metadata must be an object")
    return {"metadata_sha256": sha_file(path),
            "private_requested": data.get("is_private") is True,
            "internet_disabled_requested": data.get("enable_internet") is False,
            "gpu_disabled_requested": data.get("enable_gpu") is False,
            "no_attached_datasets_requested": data.get("dataset_sources") == [],
            "no_attached_competitions_requested": data.get("competition_sources") == [],
            "no_attached_kernels_requested": data.get("kernel_sources") == []}


def probe(metadata_path: Path, *, connect=None, environ=None, home=None) -> dict:
    declared = requested_metadata(metadata_path)
    if not all(value for key, value in declared.items() if key != "metadata_sha256"):
        raise ValueError("Probe kernel metadata must request private offline CPU without inputs")
    env = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    secret_name = re.compile(r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|ACCESS_KEY|PRIVATE_KEY|KAGGLE_KEY)", re.I)
    credential_env_present = any(secret_name.search(name) for name in env)
    credential_file_present = (home / ".kaggle" / "kaggle.json").exists()
    drive_mount_present = any(Path(path).exists() for path in ("/content/drive", "/kaggle/working/MyDrive", "/mnt/drive"))
    if connect is None:
        def connect():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                return sock.connect_ex(("1.1.1.1", 443)) == 0
    try:
        external_tcp_succeeded = bool(connect())
        network_probe_completed = True
    except (OSError, TimeoutError):
        external_tcp_succeeded = False
        network_probe_completed = True
    return {"schema": "geocpu_kaggle_cpu_isolation_probe_v1", "metadata": declared,
            "observed": {"linux": sys.platform == "linux",
                         "process_unprivileged": hasattr(os, "geteuid") and os.geteuid() != 0,
                         "drive_mount_present": drive_mount_present,
                         "credential_named_environment_present": credential_env_present,
                         "kaggle_credential_file_present": credential_file_present,
                         "external_tcp_probe_completed": network_probe_completed,
                         "external_tcp_connect_succeeded": external_tcp_succeeded},
            "limits": {"network_probe_only_one_address": True,
                       "absence_of_credentials_proven": False,
                       "all_network_destinations_blocked_proven": False,
                       "security_isolation_proven": False,
                       "stage_a_admissible": False}}


def main() -> None:
    metadata = Path(__file__).with_name("probe-kernel-metadata.json")
    report = probe(metadata)
    output = Path("/kaggle/working/probe_status.json")
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"probe_status_sha256": sha_file(output), "stage_a_admissible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
