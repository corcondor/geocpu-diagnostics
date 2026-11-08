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

# ---------------------------------------------------------------------------
# Durable, SHA-bound cache of the rebuilt datasets (Google Drive). Every Colab
# run used to rebuild the 30500 and 40500 parents from the original sources
# (several minutes) because v47's cache lookup only searches /tmp names this
# launcher never creates. A cached file is used only after its SHA256 equals
# the pinned value, so the cache can never change what is trained.
# ---------------------------------------------------------------------------
def digest_prefix(path,nbytes):
    h=hashlib.sha256()
    left=int(nbytes)
    with open(path,"rb") as stream:
        while left:
            block=stream.read(min(1<<20,left))
            if not block:
                raise ValueError("file shorter than the inherited parent: "+str(path))
            h.update(block)
            left-=len(block)
    return h.hexdigest()

def cache_fetch(cache,name,expected,destination):
    if cache is None or not (cache/name).is_file():
        return None
    part=destination.with_name(destination.name+".part")
    shutil.copyfile(cache/name,part)
    if digest(part)!=expected:
        part.unlink()
        print("V48_CACHE_REJECTED "+json.dumps({"file":name,"reason":"sha256 mismatch"}),flush=True)
        return None
    os.replace(part,destination)
    print("V48_CACHE_HIT "+json.dumps({"file":name,"sha256":expected}),flush=True)
    return destination

def cache_store(cache,name,expected,source_file):
    if cache is None:
        return
    target=cache/name
    if target.is_file() and target.stat().st_size==source_file.stat().st_size and digest(target)==expected:
        return
    cache.mkdir(parents=True,exist_ok=True)
    part=cache/(name+f".part{os.getpid()}")
    shutil.copyfile(source_file,part)
    if digest(part)!=expected:
        part.unlink()
        raise ValueError("cache copy corrupted: "+name)
    os.replace(part,target)
    print("V48_CACHE_STORED "+json.dumps({"file":name,"sha256":expected}),flush=True)

# A100 is the target. The 30500->40500 phase ran on "NVIDIA A100-SXM4-40GB" with
# the unmodified BF16 runtime and automatic chunking (36,864 tokens fit in one
# chunk; OOM halves it and redoes the same step). Measured: 10,000 updates in
# 7,848 s including the frozen teacher, i.e. 1.27 updates/s.
GPU_PROFILES={
    "a100":{"names":("A100",),"min_vram_gib":39,"chunk_tokens":None,"updates_per_second":1.27},
    "t4":{"names":("T4",),"min_vram_gib":14,"chunk_tokens":16384,"updates_per_second":None},
}

def a100_resume_patch(text):
    """The pinned runtime refuses any GPU-name change on exact resume. Colab hands
    out A100 40GB or 80GB; both run the same BF16 kernels, so only inside a checked
    new-dataset phase an A100->A100 variant change is allowed (anything else still fails)."""
    before=('            if state["gpu_name"] != (torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"):\n'
            '                raise RuntimeError("GPU model changed on exact resume")\n')
    after=('            _v48_gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"\n'
           '            if state["gpu_name"] != _v48_gpu:\n'
           '                if not (args.new_dataset_sampler and "A100" in state["gpu_name"] and "A100" in _v48_gpu):\n'
           '                    raise RuntimeError("GPU model changed on exact resume")\n'
           '                print(f"[A100] declared variant change in new dataset phase: {state[\'gpu_name\']} -> {_v48_gpu}", flush=True)\n')
    if text.count(before)!=1:
        raise ValueError("A100 resume anchor changed in pinned runtime")
    return text.replace(before,after,1)

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

def count_rows(path):
    with open(path,"rb") as stream:
        return sum(1 for line in stream if line.strip())

def compose_40500(study,scratch,oldroot,oldtrain,prep,prepared,cache=None):
    if prepared is None:
        prepared=cache_fetch(cache,"train_30500.jsonl",oldtrain.PARENT_TRAIN_SHA,scratch/"train_30500_cached.jsonl")
    parent30500,source,corpus,original,old,runner=oldtrain.make_parent_dataset(
        study,scratch,prepared)
    cache_store(cache,"train_30500.jsonl",oldtrain.PARENT_TRAIN_SHA,parent30500)
    heldouts=oldtrain.fixed_heldouts(oldroot)
    cached=scratch/"train_40500_exact.jsonl"
    if cache_fetch(cache,"train_40500_exact.jsonl",PINNED_PARENT_TRAIN,cached) is None:
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
        cache_store(cache,"train_40500_exact.jsonl",PINNED_PARENT_TRAIN,cached)
    # Inherited assets: every one of the 423,348 parent rows, byte for byte.
    if count_rows(cached)!=423348 or digest(cached)!=PINNED_PARENT_TRAIN:
        raise ValueError("40500 parent must be the exact 423,348-row pinned file")
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
    # The candidate is the parent file verbatim followed by appended rows: prove it.
    if digest_prefix(train,parent_dataset.stat().st_size)!=PINNED_PARENT_TRAIN:
        raise ValueError("candidate does not start with the exact 423,348-row parent")
    # Earlier trainer preview uses group and origin exactly as the training runtime.
    prep.NEW_GROUP=V48_GROUP
    return train,{"input_sha256":digest(train),
                  "inherited_rows":423348,"inherited_sha256":PINNED_PARENT_TRAIN,
                  "inherited_prefix_verified":True,
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

def parity_with_receipt(prep,snapshots,train,compare_steps,cache,common):
    """Run the frozen v47 oracle once per exact (data, simulator, oracle) triple;
    later runs reuse the SHA-bound receipt instead of a second full simulation."""
    from multibudget_preflight import assert_preview_parity
    if cache is None:
        return assert_preview_parity(prep,snapshots,train,compare_steps=compare_steps,**common)
    key={"train_sha256":digest(train),
         "simulator_sha256":digest(ROOT/"multibudget_preflight.py"),
         "oracle_sha256":digest(Path(prep.__file__)),
         "compare_steps":compare_steps,
         "snapshot_sha256":hashlib.sha256(json.dumps(snapshots[compare_steps],sort_keys=True).encode()).hexdigest()}
    receipt=cache/f"parity_{key['train_sha256'][:16]}_{compare_steps}.json"
    if receipt.is_file() and load_json(receipt)=={**key,"identical":True}:
        print("V48_PARITY_RECEIPT_REUSED "+json.dumps(key),flush=True)
        return True
    assert_preview_parity(prep,snapshots,train,compare_steps=compare_steps,**common)
    cache.mkdir(parents=True,exist_ok=True)
    part=receipt.with_name(receipt.name+f".part{os.getpid()}")
    part.write_text(json.dumps({**key,"identical":True},indent=2),encoding="utf-8")
    os.replace(part,receipt)
    return True


def cpu_plan_with_replay_floor_repair(train, scratch, prep, source, original, reference,
                                      minimum_updates, maximum_updates, cache=None, repair_horizon=20000,
                                      alternatives=None):
    """Bounded CPU repair. One sampler pass per attempt covers every budget and stops
    at the first one where all old-source floors and the new-domain balance hold.

    The plan with the fewest GPU updates is chosen: if no budget up to repair_horizon
    passes without replay, the replay weights are sized from the deficits AT the
    horizon (not at the scan ceiling, which would ask for small weights but 3x more
    updates). `alternatives` (a dict) receives the first passing budget without any
    replay, or None, so the report can show the cost/repetition trade-off."""
    from replay_floor_repair import next_weights, write_weighted_candidate
    from multibudget_preflight import preview_budgets

    base_train = train
    weights = {}
    budgets = list(range(minimum_updates,maximum_updates+1,1000))
    horizon = max(b for b in budgets if b <= max(minimum_updates, min(repair_horizon, maximum_updates)))
    latest_review = None
    latest_stats = {"replay_replicas_added":0, "source_training_weights":{}}
    common = dict(meta=source/"base_metadata.json",
                  normalizer_source=source/"qwenputer_controller_v30.py",
                  base_sampler_source=source/"runtime_cotraining_v41_247M.py",
                  runtime_source=original/"training/runtime_30500.py")
    accept = lambda snap: bool(snap["passed"] and assess_exposure(reference,snap)["passed"])
    for attempt in range(5):
        if weights:
            train = scratch/f"train_v48_replay_reweighted_{attempt}.jsonl"
            latest_stats = write_weighted_candidate(base_train,train,weights)
        else:
            train = base_train
        snapshots = preview_budgets(prep,train,budgets=budgets,repeat=1,min_new=1000,accept=accept,**common)
        if attempt == 0:
            # Actual original private dataset, tested against untouched v47 oracle.
            parity_with_receipt(prep,snapshots,train,budgets[0],cache,common)
        reached = sorted(snapshots)
        reviews = {steps:assess_exposure(reference,snapshots[steps]) for steps in reached}
        passing = [steps for steps in reached
                   if snapshots[steps]["passed"] and reviews[steps]["passed"]]
        latest_review = reviews[horizon] if horizon in reviews else reviews[reached[-1]]
        if attempt == 0 and alternatives is not None:
            alternatives["first_passing_updates_without_replay"] = passing[0] if passing else None
            alternatives["scan_ceiling"] = budgets[-1]
        print("V48_CPU_REPLAY_FLOOR_AUDIT "+json.dumps({
            "attempt":attempt,"steps_scanned":[reached[0],reached[-1]],
            "passing_steps":passing,"repair_horizon":horizon,
            "horizon_old_source_declines":latest_review["old_source_declines"],
            "new_source_supervised_tokens":latest_review["new_source_supervised_tokens"],
            "replay_weights":weights,
            "replay_replica_rows":latest_stats["replay_replicas_added"],
            "gpu_updates":0},ensure_ascii=False),flush=True)
        if passing and (weights or passing[0] <= horizon):
            steps=passing[0]
            return train,steps,snapshots[steps],reviews[steps],latest_stats
        if not latest_review["old_source_declines"]:
            if passing:   # beyond the horizon and nothing replay could improve: more updates are the only way
                steps=passing[0]
                return train,steps,snapshots[steps],reviews[steps],latest_stats
            raise RuntimeError(
                "Old source floors pass but new domain balance fails; "
                "do not use GPU until curriculum is fixed")
        if attempt==4:
            break
        weights=next_weights(latest_review["old_source_declines"],weights)
    raise RuntimeError("CPU replay balancing exhausted; GPU has NOT started. Remaining deficits: "+
                       json.dumps(latest_review["old_source_declines"],ensure_ascii=False))


def runtime_text(oldroot,profile="a100"):
    source_runtime=require_file(oldroot/"runtime_practical_v1.py",PINNED_RUNTIME)
    text=source_runtime.read_text(encoding="utf-8")
    if text.count('"practical_multistep_v1"')<5:
        raise ValueError("runtime group replacement sites changed")
    patched=text.replace('"practical_multistep_v1"',f'"{V48_GROUP}"')
    patched=patched.replace("KL(parent30500/replay)","KL(parent40500/replay)")
    if profile=="a100":
        # Same BF16 numerics and automatic chunking as the 30500->40500 A100 phase.
        patched=a100_resume_patch(patched)
    elif profile=="t4":
        # 16-GiB T4: fp16 with loss scaling, activation checkpointing, guarded A100->T4.
        from t4_runtime_patch import patch_runtime
        patched=patch_runtime(patched)
    else:
        raise ValueError("unknown GPU profile: "+str(profile))
    ast.parse(patched)
    return patched

def patched_runtime(oldroot,source,profile="a100"):
    runtime=source/"runtime_v48_40500.py"
    runtime.write_text(runtime_text(oldroot,profile),encoding="utf-8")
    return runtime

def command_for(source,local,corpus,train,runtime,output,stop,profile="a100"):
    chunk=GPU_PROFILES[profile]["chunk_tokens"]
    return [sys.executable,"-u",str(runtime),
      "--train_path",str(train),"--meta_path",str(source/"base_metadata.json"),
      "--corpus_path",str(corpus),
      "--warmstart_path",str(local/"checkpoint_step_40500.pt"),
      "--required_start_step","40500",
      "--resume_state",str(local/"train_state_step_40500.pt"),
      "--output_dir",str(local),
      "--max_steps",str(stop),"--stop_at_step",str(stop),
      "--batch_size","64","--step_tokens","36864",
      *(["--chunk_tokens",str(chunk)] if chunk else []),
      "--phase_lr","3e-6","--phase_floor","3e-6",
      "--phase_start","40500","--phase_warmup","0",
      "--skip_final_eval","--mask_tool_output","--save_optimizer","--numina_repeat","1",
      "--new_dataset_sampler","--new_curriculum_repeat","1",
      "--new_group_min_exposures","1000",
      "--distill_teacher",str(local/"checkpoint_step_40500.pt"),
      "--distill_weight","1.0","--distill_temperature","1.0",
      "--practical_variant","multistep","--min_replay_share","0.75",
      "--telemetry_jsonl",str(output/"training_events.jsonl")]

def source_floor_table(reference,preview):
    """Per old source: absolute supervised tokens in the 30500->40500 window vs the plan."""
    before=reference["source_supervised_tokens"]
    after=preview["source_supervised_tokens"]
    return {k:{"reference_40500_window":v,"planned":int(after.get(k,0)),
               "ratio":round(after.get(k,0)/v,4) if v else None} for k,v in sorted(before.items())}

def estimated_gpu_hours(profile,steps):
    rate=GPU_PROFILES[profile]["updates_per_second"]
    return None if not rate else round(steps/rate/3600,2)

def validate_approval(approval,*,runtime_sha,profile,metadata,minimum,maximum,confirm):
    """GPU side: accept only a CPU report produced by this revision, for the same
    weights, runtime bytes, GPU profile and curriculum; the caller must restate the
    approved update count (explicit acceptance of the GPU time it implies)."""
    if (approval.get("event")!="v48_preflight"
        or approval.get("preflight_revision")!="cpu_fast_v2"
        or approval.get("start_step")!=40500
        or approval.get("parent_checkpoint_sha256")!=PINNED_PARENT_MODEL
        or approval.get("parent_train_sha256")!=PINNED_PARENT_TRAIN
        or approval.get("runtime_sha256")!=runtime_sha
        or approval.get("gpu_profile")!=profile):
        raise ValueError("CPU approval is missing, stale, or bound to different code/weights/GPU profile")
    prior=approval.get("train",{})
    if (prior.get("new_unique_examples")!=4096
        or prior.get("inherited_rows")!=423348
        or prior.get("repeat_factor_by_domain")!=metadata["repeat_factor_by_domain"]
        or prior.get("reference_target_tokens_by_domain")!=metadata["reference_target_tokens_by_domain"]):
        raise ValueError("Candidate curriculum differs from CPU approval")
    steps=int(approval["steps"])
    if not minimum<=steps<=maximum or steps%1000:
        raise ValueError("CPU-approved update budget is outside the caller limits")
    if confirm!=steps:
        raise ValueError(f"--confirm-updates must restate the CPU-approved {steps} updates "
                         f"(about {approval.get('estimated_gpu_hours')} GPU hours) before GPU time is spent")
    return prior,steps

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--v47-zip",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepared-30500",type=Path)
    parser.add_argument("--minimum-updates",type=int,default=10000)
    parser.add_argument("--maximum-updates",type=int,default=60000,
                        help="scan ceiling only; the plan uses the FIRST budget where every old source keeps its floor")
    parser.add_argument("--gpu-profile",choices=sorted(GPU_PROFILES),default="a100")
    parser.add_argument("--dataset-cache",type=Path,
                        help="SHA-bound cache of rebuilt parents (default: STUDY/_v48_dataset_cache)")
    parser.add_argument("--no-dataset-cache",action="store_true")
    parser.add_argument("--preflight-report",type=Path)
    parser.add_argument("--approved-preflight",type=Path,
                        help="CPU-only approved manifest from the SAME pinned input; skips the sampler preview on GPU")
    parser.add_argument("--confirm-updates",type=int,
                        help="with --execute: restate the CPU-approved update count")
    parser.add_argument("--execute",action="store_true",help="Only this flag permits GPU training")
    args=parser.parse_args()
    if args.minimum_updates<1000 or args.maximum_updates<args.minimum_updates or args.maximum_updates>100000:
        parser.error("updates must be between 1000 and 100000, with maximum>=minimum")
    if args.execute and not args.approved_preflight:
        parser.error("--execute requires --approved-preflight from the CPU stage (never debug on paid GPU)")
    study=args.study.resolve()
    output=args.output.resolve()
    if output.exists() or output==study or output.is_relative_to(study/PINNED_PARENT_RUN):
        raise FileExistsError("Output must be a new branch directory; never overwrite parent")
    cache=None if args.no_dataset_cache else (args.dataset_cache or study/"_v48_dataset_cache").resolve()
    parent,receipt=parent_receipt(study)
    scratch=Path(tempfile.mkdtemp(prefix="geocpu_v48_40500_"))
    oldroot,oldtrain,prep=unpack_v47(args.v47_zip,scratch)
    # Existing 30500 historical preparation code (including data hashes) is reused.
    parent_data,source,corpus,original,old,runner,heldouts=compose_40500(
        study,scratch,oldroot,oldtrain,prep,args.prepared_30500,cache)
    train,metadata=build_v48_dataset(parent_data,scratch,oldroot,source,prep,heldouts)
    runtime=patched_runtime(oldroot,source,args.gpu_profile)
    comparison=reference_exposure(parent)
    if args.approved_preflight:
        # The CPU phase has already checked exact supervised-token exposure.
        # The GPU phase independently rebuilds the candidate, verifies its
        # SHA256, old 40500 parent, patched runtime and the frozen source floors.
        from replay_floor_repair import write_weighted_candidate
        approval=load_json(require_file(args.approved_preflight))
        prior,steps=validate_approval(approval,runtime_sha=digest(runtime),profile=args.gpu_profile,
            metadata=metadata,minimum=args.minimum_updates,maximum=args.maximum_updates,
            confirm=args.confirm_updates if args.execute else int(approval.get("steps",-1)))
        weights=prior.get("replay_source_training_weights",{})
        if weights:
            repaired=scratch/"train_v48_approved_replay.jsonl"
            repair=write_weighted_candidate(train,repaired,weights)
            train=repaired
        else:
            repair={"replay_replicas_added":0,"source_training_weights":{}}
        metadata["input_sha256"]=digest(train)
        metadata["replay_floor_replica_rows"]=repair["replay_replicas_added"]
        metadata["replay_source_training_weights"]=repair["source_training_weights"]
        metadata["total_training_rows"]=423348+metadata["rows_with_training_weights"]+repair["replay_replicas_added"]
        if (metadata["input_sha256"]!=prior.get("input_sha256")
            or metadata["replay_floor_replica_rows"]!=prior.get("replay_floor_replica_rows")
            or metadata["total_training_rows"]!=prior.get("total_training_rows")):
            raise ValueError("Prepared training bytes differ from approved CPU preflight")
        preview=approval["sampler_preview"]
        review=assess_exposure(comparison,preview)
        if not (review["passed"] and approval.get("exposure_gate")==review
                and preview.get("steps")==steps and preview.get("passed")):
            raise ValueError("CPU-approved source floors or sampling contract do not match")
        print("V48_APPROVED_CPU_PREFLIGHT_REUSED "+json.dumps({
            "steps":steps,"training_sha256":metadata["input_sha256"],
            "gpu_updates":0,"replay_weights":weights},ensure_ascii=False),flush=True)
    else:
        alternatives={}
        train,steps,preview,review,repair=cpu_plan_with_replay_floor_repair(
            train,scratch,prep,source,original,comparison,
            args.minimum_updates,args.maximum_updates,cache,alternatives=alternatives)
        metadata["plan_alternatives"]={**alternatives,
            "estimated_gpu_hours_without_replay":(estimated_gpu_hours(args.gpu_profile,alternatives["first_passing_updates_without_replay"])
                                                  if alternatives.get("first_passing_updates_without_replay") else None)}
        metadata["input_sha256"]=digest(train)
        metadata["replay_floor_replica_rows"]=repair["replay_replicas_added"]
        metadata["replay_source_training_weights"]=repair["source_training_weights"]
        metadata["total_training_rows"]=(
            423348+metadata["rows_with_training_weights"]
            +repair["replay_replicas_added"])
        if digest_prefix(train,parent_data.stat().st_size)!=PINNED_PARENT_TRAIN:
            raise ValueError("planned training file does not start with the exact inherited parent")
    stop=PINNED_STEP+steps
    profile=GPU_PROFILES[args.gpu_profile]
    report={"event":"v48_preflight","preflight_revision":"cpu_fast_v2","model_updates":0,
       "parent_run":PINNED_PARENT_RUN,"parent_checkpoint_sha256":PINNED_PARENT_MODEL,
       "parent_train_sha256":PINNED_PARENT_TRAIN,"train":metadata,
       "records":{"inherited_rows":423348,"new_unique_examples":metadata["new_unique_examples"],
                  "new_domain_repeat_rows":metadata["rows_with_training_weights"]-metadata["new_unique_examples"],
                  "old_source_replay_replica_rows":metadata["replay_floor_replica_rows"],
                  "training_file_rows":metadata["total_training_rows"],
                  "independent_problems":423348+metadata["new_unique_examples"],
                  "note":"repeat/replica rows are explicit sampling weights, not independent problems"},
       "steps":steps,"start_step":40500,"stop_step":stop,
       "sampler_preview":preview,"exposure_gate":review,
       "old_source_floors":source_floor_table(comparison,preview),
       "runtime_sha256":digest(runtime),"gpu_profile":args.gpu_profile,
       "estimated_gpu_hours":estimated_gpu_hours(args.gpu_profile,steps),
       "gpu_training_started":False,"final_benchmark_status":"not_executed",
       "logical_step_token_budget":36864,
       "chunk_tokens":profile["chunk_tokens"] or "automatic (36,864 on A100 40GB; OOM halves and redoes the step)",
       "numerics_note":("BF16 autocast as in the 30500->40500 A100 phase" if args.gpu_profile=="a100"
                        else "FP16 autocast + dynamic loss scaling; no skipped optimizer updates permitted")}
    if args.preflight_report:
        new_json(args.preflight_report,report)
    print("V48_PREFLIGHT_PASS "+json.dumps({"start":40500,"stop":stop,
        "steps":steps,"estimated_gpu_hours":report["estimated_gpu_hours"],"gpu_profile":args.gpu_profile,
        "records":report["records"],"new_sources":review["new_source_supervised_tokens"],
        "old_declines":review["old_source_declines"]},ensure_ascii=False),flush=True)
    if not args.execute:
        return
    oldtrain.refuse_active_work(study)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU is required for --execute; CPU-only runtime cannot train this model')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    if (not any(n in gpu_name for n in profile["names"])
            or gpu_bytes < profile["min_vram_gib"] * (1024 ** 3)):
        raise RuntimeError(f'GPU profile {args.gpu_profile} needs {profile["names"]} with >= {profile["min_vram_gib"]} GiB; '
                           f'detected {gpu_name}, {gpu_bytes/2**30:.1f} GiB VRAM')
    print(f'[{args.gpu_profile.upper()}] GPU {gpu_name} / VRAM {gpu_bytes/2**30:.1f} GiB', flush=True)
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
         "output_sha256":metadata["input_sha256"],"output_rows":metadata["total_training_rows"],
         "replay_replica_rows":metadata["replay_floor_replica_rows"],
         "replay_source_weights":metadata["replay_source_training_weights"],
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
            command_for(source,local,corpus,train,runtime,output,stop,args.gpu_profile),output/"console",cwd=source)
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
