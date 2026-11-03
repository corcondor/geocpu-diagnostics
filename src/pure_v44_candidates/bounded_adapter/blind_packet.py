"""Create arm-hidden answer packets and seal grading before role lookup.

The grader receives packet.jsonl only. The role map is kept separately and
the grade pin command never opens it. This is an operational blind protocol;
the person operating the files must keep the role map from the grader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

from adapter import file_sha256
from admission import verify_dataset


PACKET_SCHEMA = "geocpu_bounded_adapter_blind_packet_v1"
GRADE_SCHEMA = "geocpu_bounded_adapter_blind_grades_v1"
PIN_SCHEMA = "geocpu_bounded_adapter_grade_pin_v1"


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        if isinstance(value, list):
            for item in value:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")


def create_packet(*, data_dir, data_manifest_sha256, paired_dir, output_dir):
    data_dir, paired_dir, out = Path(data_dir), Path(paired_dir), Path(output_dir)
    verify_dataset(data_dir, data_manifest_sha256)
    result = json.loads((paired_dir / "paired_result.json").read_text(encoding="utf-8"))
    raw_path = paired_dir / "paired_raw_answers.jsonl"
    if (result.get("status") != "COMPLETE" or result.get("raw_answer_sha256") != file_sha256(raw_path)
            or result.get("data_manifest_sha256") != data_manifest_sha256):
        raise ValueError("Paired raw answers are not complete and sealed")
    raw = _read_jsonl(raw_path)
    cases = {row["id"]: row for split in ("old_skill_eval", "unseen_eval")
             for row in _read_jsonl(data_dir / (split + ".jsonl"))}
    if len(raw) != 2 * len(cases):
        raise ValueError("Paired answer coverage differs")
    records, roles, keys = [], [], set()
    for answer in raw:
        key = (answer["arm"], answer["split"], answer["id"])
        if key in keys or answer["id"] not in cases or answer.get("status") != "observed":
            raise ValueError("Duplicate or failed raw answer")
        keys.add(key)
        case = cases[answer["id"]]
        anon_id = secrets.token_hex(16)
        records.append({"anonymous_id": anon_id, "prompt": case["prompt"],
                        "expected_answer": case["expected_answer"],
                        "raw_output": answer["raw_output"],
                        "output_sha256": answer["output_sha256"],
                        "ended_with_eos": answer["ended_with_eos"],
                        "tool_trace": answer.get("tool_trace")})
        roles.append({"anonymous_id": anon_id, "arm": answer["arm"],
                      "split": answer["split"], "id": answer["id"],
                      "output_sha256": answer["output_sha256"]})
    secrets.SystemRandom().shuffle(records)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    _write_new(out / "packet.jsonl", records)
    _write_new(out / "rolemap.jsonl", roles)
    manifest = {"schema": PACKET_SCHEMA, "raw_answer_sha256": result["raw_answer_sha256"],
                "data_manifest_sha256": data_manifest_sha256,
                "packet_sha256": file_sha256(out / "packet.jsonl"),
                "rolemap_sha256": file_sha256(out / "rolemap.jsonl"),
                "records": len(records)}
    _write_new(out / "packet_manifest.json", manifest)
    return manifest


def pin_grades(*, packet_dir, grades_path, output_path):
    """Pin grades using only the arm-hidden packet, never opening rolemap."""
    packet_dir = Path(packet_dir)
    manifest = json.loads((packet_dir / "packet_manifest.json").read_text(encoding="utf-8"))
    packet_path = packet_dir / "packet.jsonl"
    if manifest.get("schema") != PACKET_SCHEMA or file_sha256(packet_path) != manifest.get("packet_sha256"):
        raise ValueError("Blind packet differs")
    packet = {row["anonymous_id"]: row for row in _read_jsonl(packet_path)}
    grades = json.loads(Path(grades_path).read_text(encoding="utf-8"))
    if grades.get("schema") != GRADE_SCHEMA or not isinstance(grades.get("grader_id"), str):
        raise ValueError("Unexpected blind grade schema")
    seen = set()
    for row in grades.get("records", []):
        ident = row.get("anonymous_id")
        if ident in seen or ident not in packet or row.get("output_sha256") != packet[ident]["output_sha256"]:
            raise ValueError("Grade ID or raw-answer SHA differs")
        seen.add(ident)
        if type(row.get("correct")) is not bool or not isinstance(row.get("reason"), str) or not row["reason"]:
            raise ValueError("Grade correctness or reason missing")
        for axis in ("japanese_fluency", "reasoning", "repetition_control"):
            if type(row.get(axis)) is not int or row[axis] not in (0, 1, 2):
                raise ValueError("Grade axis out of range")
    if seen != set(packet):
        raise ValueError("Blind grades do not cover every packet answer")
    pin = {"schema": PIN_SCHEMA, "grade_sha256": file_sha256(grades_path),
           "packet_manifest_sha256": file_sha256(packet_dir / "packet_manifest.json"),
           "packet_sha256": manifest["packet_sha256"], "raw_answer_sha256": manifest["raw_answer_sha256"],
           "records": len(seen), "rolemap_opened_by_pin": False}
    _write_new(output_path, pin)
    return pin


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    for name in ("data_dir", "data_manifest_sha256", "paired_dir", "output_dir"):
        create.add_argument("--" + name.replace("_", "-"), required=True)
    pin = sub.add_parser("pin")
    for name in ("packet_dir", "grades_path", "output_path"):
        pin.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        print(json.dumps(create_packet(data_dir=args.data_dir,
                                       data_manifest_sha256=args.data_manifest_sha256,
                                       paired_dir=args.paired_dir,
                                       output_dir=args.output_dir), ensure_ascii=False))
    else:
        print(json.dumps(pin_grades(packet_dir=args.packet_dir, grades_path=args.grades_path,
                                    output_path=args.output_path), ensure_ascii=False))


if __name__ == "__main__":
    main()
