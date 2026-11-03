"""Cheap full-SWE census; this is not CPU/Python execution verification."""

import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

from src.pure_v44_candidates.v43_alu_to_python.convert_sample import (
    TRAIN_MEMBER, evaluation_prompts, normalized_prompt, Reject,
)
from src.pure_v44_candidates.v43_swe_to_python.convert_swe import (
    SOURCE_FAMILIES, prompt_program, task_result, parse_swe_suffix,
    validate_r0_writes, expected_commands, M,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--zip-path",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    counts=collections.Counter();by_source=collections.defaultdict(collections.Counter)
    with zipfile.ZipFile(args.zip_path) as archive:
        eval_exact,eval_norm,_=evaluation_prompts(archive)
        with archive.open(TRAIN_MEMBER) as lines:
            for raw in lines:
                row=json.loads(raw);source=row.get("source")
                if source not in SOURCE_FAMILIES:continue
                counts["swe_rows"]+=1;by_source[source]["total"]+=1
                prompt,target,full=row.get("prompt"),row.get("target"),row.get("full_text")
                if prompt in eval_exact or (isinstance(prompt,str) and normalized_prompt(prompt) in eval_norm):
                    reason="eval_prompt_excluded"
                else:
                    try:
                        if not all(isinstance(x,str) for x in (prompt,target,full)) or full not in (prompt+target,prompt+" "+target):
                            raise Reject("unverified_prompt_boundary")
                        code=prompt_program(prompt)
                        final,values=task_result(source,code)
                        pairs,stated=parse_swe_suffix(full[len(prompt):])
                        if final!=stated or stated<0 or stated>M:raise Reject("task_final_mismatch")
                        actual=[p.group(0).split("=>",1)[0].rstrip() for p in pairs]
                        if actual!=expected_commands(source,code):raise Reject("task_instruction_trace_mismatch")
                        validate_r0_writes(source,pairs,values)
                    except Reject as exc:reason=str(exc)
                    else:reason="syntactic_candidate_unverified"
                counts[reason]+=1;by_source[source][reason]+=1
    result={"source_zip_sha256":hashlib.sha256(args.zip_path.read_bytes()).hexdigest(),
            "source_member":TRAIN_MEMBER,"counts":dict(counts),
            "by_source":{k:dict(v) for k,v in sorted(by_source.items())},
            "warning":"Syntactic candidates have NOT passed original CPU, Python executor, vocabulary, EOS, or 2048-position tests."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=True))


if __name__=="__main__":main()
