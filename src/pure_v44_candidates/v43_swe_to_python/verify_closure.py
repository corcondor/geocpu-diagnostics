"""Independent source/boundary/mask closure for the executed SWE output.

This checks what the converter wrote, without repeating the expensive original
CPU or Python worker execution. The converter's completed manifest is required.
"""

import argparse
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from src.pure_v44_candidates.v43_alu_to_python.convert_sample import (
    V44_DIR, TRAIN_MEMBER, evaluation_prompts, normalized_prompt,
)
from src.pure_v44_candidates.v43_swe_to_python.convert_swe import (
    SOURCE_FAMILIES, parse_swe_suffix, instruction,
)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--zip-path",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--audit-path",type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.output_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status")!="complete":raise RuntimeError("Converter not complete")
    outpath=args.output_dir/"converted_swe.jsonl"
    if sha(outpath)!=manifest["output_sha256"]:raise RuntimeError("Converted output changed")
    sys.path.insert(0,str(V44_DIR))
    from training_data_v44 import encode_supervision
    from tir_engine_v44r1 import PYTHON_BLOCK
    counts=collections.Counter();last_line=0
    with zipfile.ZipFile(args.zip_path) as archive:
        if sha(args.zip_path)!=manifest["source_zip_sha256"]:raise RuntimeError("Source ZIP changed")
        exact_eval,norm_eval,_=evaluation_prompts(archive)
        vocab=json.loads(archive.read("base_metadata.json"))["vocab"]
        with archive.open(TRAIN_MEMBER) as source, outpath.open(encoding="utf-8") as candidates:
            source_iter=enumerate(source,1)
            for converted_raw in candidates:
                row=json.loads(converted_raw)
                ref=row["source_ref"]
                if ref["kind"]!="v43_zip_member" or ref["file"]!=TRAIN_MEMBER or not isinstance(ref["line"],int):
                    raise RuntimeError("Invalid source_ref")
                if ref["line"]<=last_line:raise RuntimeError("Source lines not strictly increasing")
                while True:
                    number,raw=next(source_iter)
                    if number>=ref["line"]:break
                if number!=ref["line"]:raise RuntimeError("Source row skipped")
                if hashlib.sha256(raw).hexdigest()!=ref["raw_sha256"]:raise RuntimeError("Source raw SHA mismatch")
                original=json.loads(raw)
                if original["source"] not in SOURCE_FAMILIES or row["original_source"]!=original["source"]:
                    raise RuntimeError("Source family mismatch")
                prompt=original["prompt"]
                if row["prompt"]!=prompt or row["full_text"]!=prompt+row["target"]:
                    raise RuntimeError("Prompt/answer boundary changed")
                if prompt in exact_eval or normalized_prompt(prompt) in norm_eval:
                    raise RuntimeError("Evaluation prompt leakage")
                suffix=original["full_text"][len(prompt):]
                pairs,_=parse_swe_suffix(suffix)
                prefix=suffix[:pairs[0].start()]
                trailing=suffix[pairs[-1].end():]
                program="R=[0]*8\nM=4294967295\ndef p(i):print(f'[R{i}={R[i]}]')\n"
                program+="\n".join(instruction(pair) for pair in pairs)+"\n"
                stdout="".join(f'[{p.group("observed")}={p.group("value")}]\n' for p in pairs)
                observation="\n```output\n"+stdout+"```\n"
                expected=prefix+"\n```python\n"+program+"```\n"+observation+trailing
                if row["target"]!=expected:raise RuntimeError("Converted target differs from exact source command plan")
                native_blocks=list(PYTHON_BLOCK.finditer(row["target"]))
                if len(native_blocks)!=1 or native_blocks[0].group(1)!=program:
                    raise RuntimeError("Native v44 Python fence not detected")
                start=len(prefix+"\n```python\n"+program+"```\n")
                if row["target_segments"]!=[{"start":start,"end":start+len(observation),"role":"tool","train":False}]:
                    raise RuntimeError("Tool observation span mismatch")
                _,tags=encode_supervision(row,vocab,max_positions=2048)
                if tags[-1]!=1 or any(tags[len(prompt)+start:len(prompt)+start+len(observation)]):
                    raise RuntimeError("Tool mask or true EOS mismatch")
                counts["accepted"]+=1;counts[original["source"]]+=1;last_line=number
    if counts["accepted"]!=manifest["counts"]["accepted"]:
        raise RuntimeError("Accepted count does not match converter manifest")
    audit={"status":"PASS","source_zip_sha256":manifest["source_zip_sha256"],
           "converted_sha256":manifest["output_sha256"],"accepted":counts["accepted"],
           "accepted_by_source":{k:v for k,v in counts.items() if k!="accepted"},
           "last_source_line_1based":last_line,
           "checks":["raw source line SHA","monotonic source refs","original prompt boundary",
                     "evaluation prompt exclusion","exact generated code and tool observations",
                     "pure v44 native Python fence","pure v44 vocabulary and 2048 positions",
                     "tool output mask and true EOS"]}
    args.audit_path.parent.mkdir(parents=True,exist_ok=True)
    args.audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=True))


if __name__=="__main__":main()
