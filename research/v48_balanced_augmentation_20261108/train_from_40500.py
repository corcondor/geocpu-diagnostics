#!/usr/bin/env python3
"""GeoCPU v48 additive continuation from verified step 40500.

This is an experimental launcher, deliberately gated by a full deterministic
sampler preview. It neither overwrites the 40500 parent nor skips old-group
absolute supervised-token floors. Only --execute actually trains.
Requires the original v47 10k ZIP, original full training dependencies in
Google Drive, and the v48 private GitHub checkout.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
PINNED_PARENT_RUN = "practical_20261107_10k_multistep_40500"
PINNED_STEP = 40500
PINNED_PARENT_TRAIN = "ba59c14bdd9e12a222b13aa725cd4e039b286b728ba9d1c53ff8900902e9839a"
PINNED_PARENT_MODEL = "4ab047ea92b2a0c8863c0174c18e5769847b4433102a8eb12752a42a68da9b79"
PINNED_PARENT_OPT = "6e7e61c20decc12affa7a023eea4d8e11cc372d80bbd6fb6bbd4ac5841234bf8"
PINNED_PARENT_STATE = "d1cba39e3eb8e8722e5a666100617f8364f8bcfe4f34e7a34fff2711b7e86a73"
PINNED_RUNTIME = "fc196cf9b7d22fae7f0128b5b0b1bd1e8b7630d6f9e27ba0fd310d0db49605c2"
PINNED_V47_ZIP = "939e75d61aba5cd9f271b46e3d210d42af5a0c51904b80df101dbf40072d55d5"
V48_GROUP = "v48_additive_only"
CATEGORIES = ("python", "logic", "evidence", "japanese")

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def require_file(path,expected=None):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if expected is not None and digest(path)!=expected:
        raise ValueError("SHA256 mismatch: "+str(path))
    return path

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def new_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"x",encoding="utf-8") as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2)
        stream.write("\n")

def parent_receipt(study):
    parent=study/PINNED_PARENT_RUN
    cp=parent/"checkpoints"
    saved=load_json(require_file(parent/"training_saved.json"))
    receipt=load_json(require_file(cp/"receipt_step_40500.json"))
    manifest=load_json(require_file(parent/"input_manifest.json"))
    expected={"checkpoint":PINNED_PARENT_MODEL,"optimizer":PINNED_PARENT_OPT,"state":PINNED_PARENT_STATE}
    if (saved.get("step")!=PINNED_STEP or receipt.get("step")!=PINNED_STEP
        or saved.get("training_data_sha256")!=PINNED_PARENT_TRAIN
        or receipt.get("training_data_sha256")!=PINNED_PARENT_TRAIN
        or manifest.get("output_sha256")!=PINNED_PARENT_TRAIN
        or manifest.get("output_rows")!=423348
        or not saved.get("full_model") or receipt.get("runtime_sha256")!=PINNED_RUNTIME):
        raise ValueError("40500 saved metadata or parent input does not match pinned run")
    for kind,hashvalue in expected.items():
        filename="train_state" if kind=="state" else kind
        p=cp/f"{filename}_step_{PINNED_STEP}.pt"
        if not p.is_file() or receipt.get(kind+"_sha256")!=hashvalue:
            raise ValueError("40500 checkpoint artifact or receipt is missing/mismatched: "+str(p))
    return parent,receipt

def unpack_v47(archive,scratch):
    import importlib
    if digest(archive)!=PINNED_V47_ZIP:
        raise ValueError("v47 ZIP changed")
    old=scratch/"v47"
    old.mkdir()
    with zipfile.ZipFile(archive) as z:
        if any(name.startswith("/") or ".." in Path(name).parts for name in z.namelist()):
            raise ValueError("unsafe ZIP path")
        z.extractall(old)
    sys.path.insert(0,str(old))
    importlib.invalidate_caches()
    import train_candidate as old_train
    import prepare_training as old_prep
    return old,old_train,old_prep

def compose_40500(study,scratch,oldroot,oldtrain,prep,prepared):
    parent30500,source,corpus,original,old,runner=oldtrain.make_parent_dataset(
        study,scratch,prepared)
    heldouts=oldtrain.fixed_heldouts(oldroot)
    cached=scratch/"train_40500_exact.jsonl"
    manifest=scratch/"recompose_40500.json"
    additions=[
        (oldroot/"data/practical_train.jsonl",oldroot/"data/practical_manifest.json"),
        (oldroot/"data/knowledge_train.jsonl",oldroot/"data/knowledge_manifest.json")]
    prep.prepare_dataset(parent30500,additions,meta=source/"base_metadata.json",
        normalizer_source=source/"qwenputer_controller_v30.py",
        heldouts=heldouts,output=cached,manifest_output=manifest,variant="multistep")
    require_file(cached,PINNED_PARENT_TRAIN)
    if load_json(manifest).get("output_rows")!=423348:
        raise ValueError("40500 exact parent row count mismatch")
    return cached,source,corpus,original,old,runner,heldouts

def build_v48_dataset(parent_dataset,scratch,oldroot,source,prep,heldouts):
    manifest=load_json(require_file(ROOT/"data/manifest.json"))
    normalize=prep.normalization_table(source/"qwenputer_controller_v30.py")
    vocab=set(load_json(source/"base_metadata.json")["vocab"])
    additional_heldouts=list(heldouts)
    for domain in CATEGORIES:
        for split in ("dev","unseen_eval"):
            filename=f"{domain}_{split}.jsonl"
            info=manifest["files"][filename]
            additional_heldouts.append(require_file(ROOT/"data"/filename,info["sha256"]))
    heldout_prompts,heldout_ids,heldout_hashes=prep.heldout_sets(additional_heldouts,normalize)
    existing=set()
    with parent_dataset.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                existing.add(json.loads(line)["prompt"].translate(normalize))
    base_rows={}
    used_ids=set()
    supervised_sum={}
    for domain in CATEGORIES:
        filename=f"{domain}_train.jsonl"
        info=manifest["files"][filename]
        datafile=require_file(ROOT/"data"/filename,info["sha256"])
        rows=[json.loads(line) for line in datafile.read_text(encoding="utf-8").splitlines()]
        if len(rows)!=1024 or info["rows"]!=1024:
            raise ValueError("v48 category count is not exactly 1024: "+domain)
        accepted=[]
        for row in rows:
            row=dict(row)
            row["_v46_group"]="v48_"+domain
            prep.ACCEPTED_GROUPS.add(row["_v46_group"])
            prep.validate_new_row(row,vocab,normalize,heldout_prompts,heldout_ids)
            if row["id"] in used_ids or row["prompt"].translate(normalize) in existing:
                raise ValueError("new/old exact prompt overlap or repeated ID: "+row["id"])
            used_ids.add(row["id"])
            existing.add(row["prompt"].translate(normalize))
            supervised=len(row["target"])+1-sum(b-a for a,b in row["loss_mask_spans"])
            if supervised<=0:
                raise ValueError("empty supervised target")
            accepted.append(row)
        base_rows[domain]=accepted
        supervised_sum[domain]=sum(len(row["target"])+1-sum(b-a for a,b in row["loss_mask_spans"])
                               for row in accepted)
    # Compensate for unequal text lengths: balance new source teacher-token mass,
    # not merely source row counts. Repetitions are training weights, NOT new tasks.
    max_mean=max(supervised_sum.values())
    repeats={domain:min(16,max(1,round(max_mean/supervised_sum[domain])))
             for domain in CATEGORIES}
    train=scratch/"train_40500_plus_v48.jsonl"
    shutil.copyfile(parent_dataset,train)
    with train.open("a",encoding="utf-8") as out:
        for domain in CATEGORIES:
            for row in base_rows[domain]:
                for replica in range(repeats[domain]):
                    rr=dict(row)
                    if replica:
                        rr["id"]=row["id"]+f"_training_repeat_{replica}"
                    rr["_practical_source"]="v48_"+domain
                    rr["_v46_group"]=V48_GROUP
                    rr["training_variant"]="multistep"
                    out.write(json.dumps(rr,ensure_ascii=False,separators=(",",":"))+"\n")
    # Earlier trainer preview uses group and origin exactly as the training runtime.
    prep.NEW_GROUP=V48_GROUP
    return train,{"input_sha256":digest(train),
                  "new_unique_examples":sum(map(len,base_rows.values())),
                  "rows_with_training_weights":sum(len(base_rows[d])*repeats[d] for d in CATEGORIES),
                  "repeat_factor_by_domain":repeats,
                  "reference_target_tokens_by_domain":supervised_sum,
                  "heldout_sha256":heldout_hashes,
                  "parent_rows":423348}

def reference_exposure(parent):
    path=require_file(parent/"training_events.jsonl")
    final=None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record=json.loads(line)
                if record.get("event")=="train_step":
                    final=record
    if not final or final.get("step")!=40500 or not final.get("source_supervised_tokens"):
        raise ValueError("40500 source token evidence unavailable")
    return final

def assess_exposure(reference,preview,tolerance=0.15):
    before=reference["source_supervised_tokens"]
    after=preview["source_supervised_tokens"]
    declines={k:{"baseline":v,"candidate":after.get(k,0)}
              for k,v in before.items() if after.get(k,0)<v}
    new={cat:after.get("v48_"+cat,0) for cat in CATEGORIES}
    positive=all(x>0 for x in new.values())
    balanced=positive and min(new.values())>=max(new.values())*(1-tolerance)
    return {"passed":not declines and positive and balanced,
        "old_source_declines":declines,"new_source_supervised_tokens":new,
        "new_sources_balanced_within_15pct":balanced,
        "old_source_count":len(before)}

def patched_runtime(oldroot,source):
    source_runtime=require_file(oldroot/"runtime_practical_v1.py",PINNED_RUNTIME)
    text=source_runtime.read_text(encoding="utf-8")
    if text.count('"practical_multistep_v1"')<5:
        raise ValueError("runtime group replacement sites changed")
    patched=text.replace('"practical_multistep_v1"',f'"{V48_GROUP}"')
    patched=patched.replace("KL(parent30500/replay)","KL(parent40500/replay)")
    ast.parse(patched)
    runtime=source/"runtime_v48_40500.py"
    runtime.write_text(patched,encoding="utf-8")
    return runtime

def command_for(source,local,corpus,train,runtime,output,stop):
    return [sys.executable,"-u",str(runtime),
      "--train_path",str(train),"--meta_path",str(source/"base_metadata.json"),
      "--corpus_path",str(corpus),
      "--warmstart_path",str(local/"checkpoint_step_40500.pt"),
      "--required_start_step","40500",
      "--resume_state",str(local/"train_state_step_40500.pt"),
      "--output_dir",str(local),
      "--max_steps",str(stop),"--stop_at_step",str(stop),
      "--batch_size","64","--step_tokens","36864",
      "--phase_lr","3e-6","--phase_floor","3e-6",
      "--phase_start","40500","--phase_warmup","0",
      "--skip_final_eval","--mask_tool_output","--save_optimizer","--numina_repeat","1",
      "--new_dataset_sampler","--new_curriculum_repeat","1",
      "--new_group_min_exposures","1000",
      "--distill_teacher",str(local/"checkpoint_step_40500.pt"),
      "--distill_weight","1.0","--distill_temperature","1.0",
      "--practical_variant","multistep","--min_replay_share","0.75",
      "--telemetry_jsonl",str(output/"training_events.jsonl")]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--v47-zip",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepared-30500",type=Path)
    parser.add_argument("--minimum-updates",type=int,default=10000)
    parser.add_argument("--maximum-updates",type=int,default=20000)
    parser.add_argument("--preflight-report",type=Path)
    parser.add_argument("--execute",action="store_true",help="Only this flag permits GPU training")
    args=parser.parse_args()
    if args.minimum_updates<1000 or args.maximum_updates<args.minimum_updates or args.maximum_updates>20000:
        parser.error("updates must be between 1000 and 20000, with maximum>=minimum")
    study=args.study.resolve()
    output=args.output.resolve()
    if output.exists() or output==study or output.is_relative_to(study/PINNED_PARENT_RUN):
        raise FileExistsError("Output must be a new branch directory; never overwrite parent")
    parent,receipt=parent_receipt(study)
    scratch=Path(tempfile.mkdtemp(prefix="geocpu_v48_40500_"))
    oldroot,oldtrain,prep=unpack_v47(args.v47_zip,scratch)
    # Existing 30500 historical preparation code (including data hashes) is reused.
    parent_data,source,corpus,original,old,runner,heldouts=compose_40500(
        study,scratch,oldroot,oldtrain,prep,args.prepared_30500)
    train,metadata=build_v48_dataset(parent_data,scratch,oldroot,source,prep,heldouts)
    runtime=patched_runtime(oldroot,source)
    comparison=reference_exposure(parent)
    chosen=None
    for steps in range(args.minimum_updates,args.maximum_updates+1,1000):
        preview=prep.preview_plan(train,meta=source/"base_metadata.json",
            normalizer_source=source/"qwenputer_controller_v30.py",
            base_sampler_source=source/"runtime_cotraining_v41_247M.py",
            runtime_source=original/"training/runtime_30500.py",
            steps=steps,repeat=1,min_new=1000)
        review=assess_exposure(comparison,preview)
        print("V48_EXPOSURE_PREVIEW "+json.dumps({
            "updates":steps, "passed":review["passed"],
            "old_group_deficit_count":len(review["old_source_declines"]),
            "new_tokens":review["new_source_supervised_tokens"],
            "new_balanced":review["new_sources_balanced_within_15pct"]},
            ensure_ascii=False),flush=True)
        if review["passed"]:
            chosen=(steps,preview,review)
            break
    if chosen is None:
        raise RuntimeError("No safe budget found: old sources or new domain balance insufficient. "
            "GPU training has NOT started; change curriculum weights rather than lowering floors.")
    steps,preview,review=chosen
    stop=PINNED_STEP+steps
    report={"event":"v48_preflight","model_updates":0,
       "parent_run":PINNED_PARENT_RUN,"parent_checkpoint_sha256":PINNED_PARENT_MODEL,
       "parent_train_sha256":PINNED_PARENT_TRAIN,"train":metadata,
       "steps":steps,"start_step":40500,"stop_step":stop,
       "sampler_preview":preview,"exposure_gate":review,
       "runtime_sha256":digest(runtime),"gpu_training_started":False,
       "final_benchmark_status":"not_executed"}
    if args.preflight_report:
        new_json(args.preflight_report,report)
    print("V48_PREFLIGHT_PASS "+json.dumps({"start":40500,"stop":stop,
        "steps":steps,"new_sources":review["new_source_supervised_tokens"],
        "old_declines":review["old_source_declines"]},ensure_ascii=False),flush=True)
    if not args.execute:
        return
    oldtrain.refuse_active_work(study)
    torch=oldtrain.check_a100()
    oldtrain.PARENT_STEP=40500
    oldtrain.CHECKPOINT_INTERVAL=5000
    oldtrain.PARENT_ARTIFACTS={"checkpoint":PINNED_PARENT_MODEL,"optimizer":PINNED_PARENT_OPT,"state":PINNED_PARENT_STATE}
    local=scratch/"training_checkpoints"
    local.mkdir()
    for kind,destination in (("checkpoint","checkpoint_step_40500.pt"),
                             ("optimizer","optimizer_latest.pt"),("state","train_state_step_40500.pt")):
        srcname="train_state" if kind=="state" else kind
        old.copy_verified_once(
            parent/"checkpoints"/f"{srcname}_step_40500.pt",
            local/destination,receipt[kind+"_sha256"])
    output.mkdir(parents=True,exist_ok=False)
    new_json(output/"training_active.json",
        {"pid":os.getpid(),"parent_sha256":PINNED_PARENT_MODEL,
         "start_step":40500,"stop_step":stop})
    new_json(output/"preflight.json",report)
    new_json(output/"input_manifest.json",
        {"parent_sha256":PINNED_PARENT_TRAIN,"parent_rows_preserved":423348,
         "new_unique_rows":4096,"weighted_training_rows":metadata["rows_with_training_weights"],
         "output_sha256":metadata["input_sha256"],"output_rows":423348+metadata["rows_with_training_weights"],
         "new_group":V48_GROUP,"source_repeat_weights":metadata["repeat_factor_by_domain"]})
    new_json(output/"fork_from_40500.json",
        {"step_start":40500,"step_stop":stop,"parent_sha256":PINNED_PARENT_MODEL,
         "teacher_sha256":PINNED_PARENT_MODEL,"teacher_frozen":True,"learning_rate":3e-6,
         "supervised_exposure_gate":review,"sampler_reset":True})
    shutil.copyfile(runtime,output/"runtime_v48_40500.py")
    backup=oldtrain.IntermediateCheckpointBackup(
        local=local,output=output,steps=steps,
        state_loader=lambda path:torch.load(path,map_location="cpu",weights_only=True),
        binding={"training_data_sha256":metadata["input_sha256"],"parent_sha256":PINNED_PARENT_MODEL,
                 "runtime_sha256":digest(runtime),"variant":"multistep",
                 "new_curriculum_repeat":1,"teacher_sha256":PINNED_PARENT_MODEL})
    try:
        backup.start()
        rc=oldtrain.run_with_logging(
            command_for(source,local,corpus,train,runtime,output,stop),output/"console",cwd=source)
        backup.request_stop()
        if rc:
            raise RuntimeError("GPU process exit "+str(rc)+"; see console.log")
        final_state=torch.load(local/f"train_state_step_{stop}.pt",map_location="cpu",weights_only=True)
        if final_state.get("step")!=stop:
            raise RuntimeError("Final state does not match target stop")
        actual=Counter()
        last=None
        with (output/"training_events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                e=json.loads(line)
                if e.get("event")=="train_step":last=e
        if last is None or last.get("step")!=stop:
            raise RuntimeError("No completed final model telemetry")
        actual_review=assess_exposure(comparison,last)
        if not actual_review["passed"] or last["source_supervised_tokens"]!=preview["source_supervised_tokens"]:
            raise RuntimeError("Actual source token exposure mismatches preflight; do NOT adopt checkpoint")
        final_receipt={"step":stop,"branch_from":40500,
            "training_data_sha256":metadata["input_sha256"],
            "parent_sha256":PINNED_PARENT_MODEL,
            "teacher_sha256":PINNED_PARENT_MODEL,"runtime_sha256":digest(runtime),
            "exposure_gate":actual_review,"evaluation":"pending"}
        for kind in ("checkpoint","optimizer","state"):
            name="train_state" if kind=="state" else kind
            filename=f"{name}_step_{stop}.pt"
            final_receipt[kind+"_sha256"]=runner.atomic_copy(
                local/filename,output/"checkpoints"/filename)
        new_json(output/f"checkpoints/receipt_step_{stop}.json",final_receipt)
        new_json(output/"training_saved.json",
            {"step":stop,"branch_from":40500,"full_model":True,
             "training_data_sha256":metadata["input_sha256"],
             "evaluation":"pending","exposure_gate":"passed"})
        print("V48_TRAIN_SAVED_EVAL_PENDING "+json.dumps(final_receipt,ensure_ascii=False),flush=True)
    except BaseException as error:
        new_json(output/"training_incomplete.json",
            {"error":str(error),"kind":type(error).__name__,
             "scratch_path":str(scratch)})
        raise
    finally:
        backup.request_stop()
        backup.close()
        (output/"training_active.json").unlink(missing_ok=True)

if __name__=="__main__":
    main()
