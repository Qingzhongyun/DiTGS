"""Atomic same-topology distributed checkpoints for per-view Wan.

FSDP uses SHARDED_STATE_DICT and the optimizer's actual same-rank local state. Every parameter (including frozen ones), persistent/nonpersistent buffer,
Adam state, EMA tensor and caller-provided rank RNG is retained.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from .config import pfs_path

FORMAT = "ditgs-wan-multiview-distributed-sharded"

def _rank_contract(rank, world_size, sharded):
    if not 0 <= rank < world_size:
        raise ValueError("Invalid checkpoint rank/world")
    if dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size()) != (rank, world_size):
            raise ValueError("Checkpoint rank/world differs from initialized default process group")
    elif world_size != 1 or sharded:
        raise RuntimeError("Initialize the matching process group before distributed checkpoints")

def _core(model):
    return model.module if isinstance(model, DistributedDataParallel) else model

def _state_context(model, sharded):
    if sharded:
        if not any(isinstance(module, FSDP) for module in model.modules()):
            raise ValueError("sharded=True requires an FSDP model")
        # LOCAL_STATE_DICT + use_orig_params=True leaves unexpected _flat_param
        # keys on strict load in torch 2.8. Logical per-parameter sharded states
        # support strict same-topology resume and freeze/unfreeze initialization.
        return FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                   ShardedStateDictConfig(offload_to_cpu=False))
    return nullcontext()

def _cpu_tree(value):
    if hasattr(value, "local_shards"):
        # ShardedTensor.detach() copies its metadata with requires_grad=False,
        # matching the detached local shards without changing live parameters.
        return value.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        result = type(value)((key, _cpu_tree(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = value._metadata
        return result
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value

def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}

def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _write_torch(path, value):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _digest(path)

def _write_json(path, value):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def _gather(value, world_size):
    if world_size == 1:
        return [value]
    values = [None] * world_size
    dist.all_gather_object(values, value)
    return values

def _broadcast(value, rank, world_size):
    values = [value if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(values, src=0)
    return values[0]

def _agree_errors(error, stage, world_size):
    errors = _gather(error, world_size)
    if any(item is not None for item in errors):
        detail = "; ".join(f"rank{rank}: {item}" for rank, item in enumerate(errors) if item is not None)
        raise RuntimeError(f"Distributed checkpoint {stage} failed: {detail}")

def _parameter_layout(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [{"name": group.get("name", str(index)),
             "params": [{"name": names.get(id(p), "<not-in-model>"),
                         "shape": list(p.shape), "dtype": str(p.dtype),
                         "requires_grad": p.requires_grad} for p in group["params"]]}
            for index, group in enumerate(optimizer.param_groups)]

def _buffer_state(model):
    return {name: tensor.detach().cpu() for name, tensor in model.named_buffers()}

def _restore_buffers(model, saved):
    current = dict(model.named_buffers())
    if current.keys() != saved.keys():
        raise ValueError("Checkpoint buffer topology changed")
    with torch.no_grad():
        for name, value in saved.items():
            tensor = current[name]
            if tensor.shape != value.shape or tensor.dtype != value.dtype:
                raise ValueError(f"Checkpoint buffer shape/dtype differs: {name}")
            tensor.copy_(value.to(device=tensor.device))

def _restore_ema(current, saved, prefix="ema"):
    # Copy into current tensors to retain their intended CPU/GPU placement.
    if isinstance(current, torch.Tensor):
        if not isinstance(saved, torch.Tensor) or current.shape != saved.shape or current.dtype != saved.dtype:
            raise ValueError(f"EMA shape/dtype mismatch at {prefix}")
        current.copy_(saved.to(device=current.device))
        return current
    if isinstance(current, dict):
        if not isinstance(saved, dict) or current.keys() != saved.keys():
            raise ValueError(f"EMA key mismatch at {prefix}; initialize EMA topology before restoring")
        for key in current:
            current[key] = _restore_ema(current[key], saved[key], f"{prefix}/{key}")
        return current
    if isinstance(current, (tuple, list)):
        if not isinstance(saved, type(current)) or len(current) != len(saved):
            raise ValueError(f"EMA sequence mismatch at {prefix}")
        values = [_restore_ema(old, new, f"{prefix}/{index}") for index, (old, new) in enumerate(zip(current, saved))]
        return tuple(values) if isinstance(current, tuple) else values
    return saved

def save_checkpoint(model, optimizer, ema, folder, *, rank, world_size,
                    sharded, metadata, rng_state):
    """All ranks call. Return published Path; never overwrite any existing checkpoint.

    metadata must be JSON-serializable and should include recipe_hash, completed_steps,
    update_clock and source revisions. RNG is opaque caller state, returned on load.
    """
    _rank_contract(rank, world_size, sharded)
    folder = pfs_path(str(folder), "checkpoint folder")
    core = _core(model)
    error = None
    try:
        metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        identity = {"folder": str(folder), "sharded": bool(sharded),
                    "metadata_sha256": hashlib.sha256(metadata_json.encode()).hexdigest()}
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        identity = None
    _agree_errors(error, "metadata validation", world_size)
    identities = _gather(identity, world_size)
    if any(item != identities[0] for item in identities):
        raise ValueError("Ranks disagree on checkpoint path/layout/shared metadata")

    lock = folder.with_name("." + folder.name + ".publish.lock")
    descriptor = None
    preparation = None
    if rank == 0:
        try:
            folder.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            if folder.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint {folder}")
            temporary = folder.with_name("." + folder.name + ".incomplete-" + uuid.uuid4().hex)
            temporary.mkdir()
            preparation = {"temporary": str(temporary), "error": None}
        except Exception as exception:
            preparation = {"error": f"{type(exception).__name__}: {exception}"}
    preparation = _broadcast(preparation, rank, world_size)
    try:
        if preparation["error"] is not None:
            raise RuntimeError("Checkpoint preparation failed: " + preparation["error"])
        temporary = Path(preparation["temporary"])
        receipts = {}
        error = None
        try:
            # All ranks enter FSDP's state_dict hooks; plain DDP needs only rank0.
            if sharded or rank == 0:
                with _state_context(core, sharded):
                    model_state = core.state_dict()
                state = {"format": FORMAT, "rank": rank if sharded else 0,
                         "world_size": world_size, "sharded": bool(sharded),
                         "model": _cpu_tree(model_state), "all_buffers": _buffer_state(core),
                         "optimizer": _cpu_tree(optimizer.state_dict()),
                         "optimizer_layout": _parameter_layout(core, optimizer),
                         "ema": _cpu_tree(ema)}
                filename = f"state_rank{rank:05d}.pt" if sharded else "state.pt"
                receipts[filename] = _write_torch(temporary / filename, state)
                del state, model_state
            rng_name = f"rng_rank{rank:05d}.pt"
            receipts[rng_name] = _write_torch(temporary / rng_name,
                                            {"rank": rank, "world_size": world_size,
                                             "rng_state": _cpu_tree(rng_state)})
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        _agree_errors(error, "rank state write", world_size)
        rank_receipts = _gather(receipts, world_size)
        publication_error = None
        if rank == 0:
            try:
                files = {}
                for receipt in rank_receipts:
                    if files.keys() & receipt.keys():
                        raise RuntimeError("Duplicate checkpoint filenames across ranks")
                    files.update(receipt)
                expected = {f"rng_rank{r:05d}.pt" for r in range(world_size)}
                expected |= ({f"state_rank{r:05d}.pt" for r in range(world_size)} if sharded else {"state.pt"})
                if files.keys() != expected:
                    raise RuntimeError("Incomplete rank receipts")
                complete = {"format": FORMAT, "world_size": world_size, "sharded": bool(sharded),
                            "metadata": metadata, "files": files,
                            "state_dict_type": "SHARDED_STATE_DICT" if sharded else "FULL_DDP_RANK0",
                            "optimizer_type": "rank_local_native" if sharded else "shared_native"}
                _write_json(temporary / "COMPLETE.json", complete)
                _fsync_directory(temporary)
                # The exclusive publication lock prevents two cooperating writers.
                if folder.exists():
                    raise FileExistsError(f"Checkpoint appeared during publication: {folder}")
                os.rename(temporary, folder)
                _fsync_directory(folder.parent)
            except Exception as exception:
                publication_error = f"{type(exception).__name__}: {exception}"
        publication_error = _broadcast(publication_error, rank, world_size)
        if publication_error is not None:
            raise RuntimeError("Checkpoint publication failed: " + publication_error)
        return folder
    finally:
        if rank == 0 and descriptor is not None:
            os.close(descriptor)
            lock.unlink()

def _manifest(folder, rank, world_size, sharded, expected_recipe_hash=None):
    complete = folder / "COMPLETE.json"
    if not complete.is_file():
        raise FileNotFoundError(f"Checkpoint is not complete: {folder}")
    manifest = json.loads(complete.read_text())
    if manifest.get("format") != FORMAT or manifest.get("world_size") != world_size or manifest.get("sharded") != bool(sharded):
        raise ValueError("Checkpoint format/world/sharding topology mismatch")
    if expected_recipe_hash is not None and manifest["metadata"].get("recipe_hash") != expected_recipe_hash:
        raise ValueError("Checkpoint recipe hash mismatch")
    names = [f"state_rank{rank:05d}.pt" if sharded else "state.pt", f"rng_rank{rank:05d}.pt"]
    for name in names:
        path = folder / name
        if name not in manifest["files"] or not path.is_file() or _digest(path) != manifest["files"][name]:
            raise ValueError(f"Checkpoint file missing or checksum mismatch: {name}")
    return manifest, names

def load_checkpoint(model, optimizer, ema, folder, *, rank, world_size, sharded,
                    expected_recipe_hash=None):
    """All ranks call; restore exact same-rank local states and return metadata/RNG.

    The caller restores returned rng_state after all checkpoint I/O. The same
    architecture, FSDP wrapping, optimizer parameter order and EMA topology are
    required. There is deliberately no resharding or partial strict=False load.
    """
    _rank_contract(rank, world_size, sharded)
    folder = pfs_path(str(folder), "checkpoint folder")
    error = None
    try:
        manifest, names = _manifest(folder, rank, world_size, sharded, expected_recipe_hash)
        state = torch.load(folder / names[0], map_location="cpu", weights_only=False)
        rank_rng = torch.load(folder / names[1], map_location="cpu", weights_only=False)
        if rank_rng["rank"] != rank or rank_rng["world_size"] != world_size:
            raise ValueError("RNG shard identity mismatch")
        core = _core(model)
        if state["optimizer_layout"] != _parameter_layout(core, optimizer):
            raise ValueError("Optimizer parameter order/layout changed; cannot restore local Adam")
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    _agree_errors(error, "load preflight", world_size)
    error = None
    try:
        with _state_context(core, sharded):
            core.load_state_dict(state["model"], strict=True)
        _restore_buffers(core, state["all_buffers"])
        optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            _restore_ema(ema, state["ema"])
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    _agree_errors(error, "state restore", world_size)
    return {"metadata": manifest["metadata"], "rng_state": rank_rng["rng_state"]}



def load_model_weights(model, folder, *, rank, world_size, sharded, expected_contract):
    """Cross-recipe initialization: strict model only, fresh Adam/EMA/LR/RNG."""
    _rank_contract(rank, world_size, sharded)
    folder = pfs_path(str(folder), "initialization checkpoint")
    error = None
    try:
        manifest, names = _manifest(folder, rank, world_size, sharded)
        metadata = manifest["metadata"]
        if metadata.get("unsafe_post_update_state") or metadata.get("model_contract") != expected_contract:
            raise ValueError("Unsafe or incompatible initialization checkpoint")
        state = torch.load(folder / names[0], map_location="cpu", weights_only=False)
        core = _core(model)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    _agree_errors(error, "model initialization read", world_size)
    error = None
    try:
        with _state_context(core, sharded):
            core.load_state_dict(state["model"], strict=True)
        _restore_buffers(core, state["all_buffers"])
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    _agree_errors(error, "model-only initialization", world_size)
    return metadata
