"""Prune only this isolated experiment after a replacement is fully published."""
import json
from pathlib import Path
import re
import shutil

from .launch_setup import validate_own_recovery



def prune_replaced_checkpoints(cfg, keep):
    workspace = Path(cfg.workspace)
    if workspace.is_symlink() or workspace.resolve() not in {Path(cfg.warmup_workspace),Path(cfg.warmup_workspace).parent/"joint"}:
        raise ValueError("Retention is confined to this isolated DiTGS experiment")
    workspace = workspace.resolve()
    root = workspace / "checkpoints"
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("Refusing a redirected checkpoint directory")
    keep = Path(keep)
    if keep.is_symlink() or keep.resolve().parent != root:
        raise ValueError("Replacement must be a direct, nonsymlink checkpoint child")
    keep = keep.resolve()
    validate_own_recovery(cfg, keep)
    pointer = json.loads((workspace / "latest_safe.json").read_text())
    if Path(pointer["checkpoint"]).resolve() != keep:
        raise ValueError("Latest-safe pointer must reference the published replacement")
    completion_path = workspace / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if Path(completion["checkpoint"]).resolve() != keep:
            raise ValueError("Never replace a previously declared completed final")
    removed, skipped = [], []
    for candidate in sorted(root.iterdir()):
        if candidate == keep:
            continue
        if (candidate.is_symlink() or not candidate.is_dir() or candidate.resolve().parent != root
                or not re.fullmatch(r"[0-9a-f]{12}_[0-9]{4}_(?:initial|update[0-9]+|final)", candidate.name)):
            skipped.append(candidate.name)
            continue
        # Preserve finals during an active phase. A final published just before
        # a crash (before completion.json) may be superseded by the resumed final.
        if candidate.name.endswith("_final") and not keep.name.endswith("_final"):
            skipped.append(candidate.name)
            continue
        validate_own_recovery(cfg, candidate)
        # All boundaries, metadata, shard sizes and published pointer checked above.
        shutil.rmtree(candidate)
        removed.append(candidate.name)
    with (workspace / "checkpoint_retention.jsonl").open("a") as stream:
        stream.write(json.dumps({"keep": str(keep), "removed": removed, "skipped": skipped,
            "policy": "one latest-safe recovery during training; one final after completion"}) + "\n")
    return removed
