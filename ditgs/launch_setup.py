"""Inline phase/checkpoint setup; no model preflight or admission jobs."""
import json
import os
from pathlib import Path
from .config import config_from_json

def warmup_model_contract(cfg):
    # Same topology and corrected camera policy in both stages.
    return cfg.model_contract()


def completed_warmup_checkpoint(cfg):
    completion = json.loads((Path(cfg.warmup_workspace)/"completion.json").read_text())
    if completion.get("status") != "training_complete":
        raise ValueError("The camera-fixed adapter+GS warmup is not complete")
    return str(Path(completion["checkpoint"]).resolve())


def validate_warmup_initialization(cfg, checkpoint):
    import hashlib
    checkpoint = Path(checkpoint).resolve()
    if (checkpoint.parent != Path(cfg.warmup_workspace)/"checkpoints"
            or str(checkpoint) != completed_warmup_checkpoint(cfg)):
        raise ValueError("Joint requires the final of this camera-fixed adapter+GS warmup")
    record = json.loads((checkpoint/"COMPLETE.json").read_text())
    meta = record["metadata"]
    expected = warmup_model_contract(cfg)
    if (record.get("world_size") != cfg.world_size or record.get("sharded") is not True or
            record.get("format") != "ditgs-wan-multiview-distributed-sharded" or
            meta.get("protocol") != cfg.protocol or meta.get("model_contract") != expected or
            meta.get("recipe") != "gs_warmup" or meta.get("unsafe_post_update_state") or
            meta.get("accepted_update_budget") != 10000 or
            meta.get("progress", {}).get("accepted_updates") != 10000 or
            meta.get("progress", {}).get("completed_steps", -1) < 10000 or
            meta.get("initialization_checkpoint") != ""):
        raise ValueError("Require completed camera-fixed adapter+GS warmup10000")
    original = json.loads((checkpoint.parent.parent/"launch_config.json").read_text())
    if {k: original.get(k) for k in expected} != expected:
        raise ValueError("Warmup config does not match its saved camera/model contract")
    original.pop("resume", None)
    if hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() != meta.get("recipe_hash"):
        raise ValueError("Warmup checkpoint/config recipe mismatch")
    names = {f"state_rank{i:05d}.pt" for i in range(cfg.world_size)} | {f"rng_rank{i:05d}.pt" for i in range(cfg.world_size)}
    if set(record.get("files", {})) != names:
        raise ValueError("Incomplete warmup checkpoint file manifest")
    for name, info in record["files"].items():
        p = checkpoint/name
        if not p.is_file() or p.stat().st_size != info["bytes"]:
            raise ValueError("Missing/truncated warmup checkpoint shard: "+str(p))
    return meta


def validate_own_recovery(cfg, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent != Path(cfg.workspace).resolve()/"checkpoints":
        raise ValueError("Recovery may use only this phase's own checkpoint")
    record = json.loads((checkpoint/"COMPLETE.json").read_text())
    meta = record["metadata"]
    if (record.get("world_size") != cfg.world_size or record.get("sharded") is not True or
            record.get("format") != "ditgs-wan-multiview-distributed-sharded" or
            meta.get("protocol") != cfg.protocol or meta.get("model_contract") != cfg.model_contract() or
            meta.get("recipe") != cfg.recipe or meta.get("unsafe_post_update_state") or
            meta.get("accepted_update_budget") != cfg.steps or meta.get("recipe_hash") != cfg.recipe_hash()):
        raise ValueError("Incompatible phase/model/optimizer recovery state")
    names = {f"state_rank{i:05d}.pt" for i in range(cfg.world_size)} | {f"rng_rank{i:05d}.pt" for i in range(cfg.world_size)}
    if set(record.get("files", {})) != names:
        raise ValueError("Incomplete own checkpoint manifest")
    for name, item in record["files"].items():
        path = checkpoint/name
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("Missing/truncated own checkpoint shard: "+str(path))
    return meta


def resolve_launch_config(cfg, *, workspace="", resume="", init_checkpoint=""):
    if workspace:
        cfg.workspace = str(Path(workspace).expanduser().resolve())
    folder = Path(cfg.workspace)
    previous_path = folder/"launch_config.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    if previous and (previous.get("protocol") != cfg.protocol or previous.get("recipe") != cfg.recipe):
        raise ValueError("Refusing an incompatible existing workspace")
    cfg.resume = resume or cfg.resume
    pointer = folder/"latest_safe.json"
    if not cfg.resume and pointer.is_file():
        cfg.resume = json.loads(pointer.read_text())["checkpoint"]
    cfg.init_checkpoint = init_checkpoint or cfg.init_checkpoint
    if cfg.recipe == "joint" and not cfg.init_checkpoint:
        cfg.init_checkpoint = previous.get("init_checkpoint") or completed_warmup_checkpoint(cfg)
    cfg.validate()
    if cfg.recipe == "joint":
        validate_warmup_initialization(cfg, cfg.init_checkpoint)
    if cfg.resume:
        validate_own_recovery(cfg, cfg.resume)
    return cfg


def validate_completed_phase(cfg):
    """Idempotent sequential launch: do not re-save an already completed warmup.

    Keeping its final path stable also keeps the joint initialization hash stable
    when the user resubmits the two-stage command after a joint interruption.
    """
    completion = json.loads((Path(cfg.workspace)/"completion.json").read_text())
    if (completion.get("status") != "training_complete" or
            completion.get("protocol") != cfg.protocol or
            completion.get("model_contract") != cfg.model_contract() or
            completion.get("progress", {}).get("accepted_updates") != cfg.steps or
            completion.get("lineage_accepted_updates") != cfg.lineage_offset+cfg.steps):
        raise ValueError("Existing completion belongs to an incompatible/incomplete phase")
    metadata = validate_own_recovery(cfg, completion["checkpoint"])
    if metadata["progress"]["accepted_updates"] != cfg.steps:
        raise ValueError("Completion pointer does not reference a completed checkpoint")
    return completion


def write_launch_config(cfg, *, explicit_resume=False):
    folder = Path(cfg.workspace)
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/"launch_config.json"
    temporary = path.with_name(path.name+f".{os.getpid()}.writing")
    with temporary.open("w") as f:
        json.dump(cfg.to_dict(),f,indent=2,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    temporary.replace(path)
    (folder/"launch_request.json").write_text(json.dumps({
        "recipe":cfg.recipe,"new_accepted_update_budget":cfg.steps,"resume":cfg.resume,
        "init_checkpoint":cfg.init_checkpoint,"explicit_resume":bool(explicit_resume),
        "preflight":False,"ledger":"user_managed","source_hash_gate":False,
        "asset_receipt_gate":False,"automatic_download":False},indent=2)+"\n")
    return path


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument("--workspace",default="")
    p.add_argument("--resume",default="")
    p.add_argument("--init-checkpoint",default="")
    p.add_argument("--require-complete",action="store_true")
    a=p.parse_args(argv)
    cfg=resolve_launch_config(config_from_json(a.config),workspace=a.workspace,
        resume=a.resume,init_checkpoint=a.init_checkpoint)
    if a.require_complete:
        print(json.dumps(validate_completed_phase(cfg)), flush=True)
    else:
        print(write_launch_config(cfg,explicit_resume=bool(a.resume)),flush=True)

if __name__=="__main__":
    main()
