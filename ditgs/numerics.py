"""Numerical observability and counterfactual replay for per-view Wan jobs.

All collectives run at explicit trainer boundaries; hooks record but never throw.
This module does not submit a job and makes no numerical-safety verdict.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.distributed as dist


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unserializable diagnostic value: {type(value).__name__}")


def write_json(path: str | Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return str(path)


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_value(payload), ensure_ascii=False) + "\n")
        handle.flush()


def require_pfs(path):
    """Normalize the explicitly selected experiment directory."""
    return Path(path).expanduser().resolve()


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _named_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    """Traverse local tensors, including FSDP LOCAL_STATE_DICT shard payloads."""
    if hasattr(value, "local_shards"):
        for i, shard in enumerate(value.local_shards()):
            yield f"{prefix}/shard{i}", shard.tensor
    elif isinstance(value, torch.Tensor):
        # DTensor presents itself as a Tensor; explicitly expose its local shard.
        tensor = value.to_local() if hasattr(value, "to_local") else value
        yield prefix, tensor
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _named_tensors(item, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _named_tensors(item, f"{prefix}/{i}")


def tensor_statistics(tensor: torch.Tensor, chunk_elements: int = 1_048_576) -> dict[str, Any]:
    """Stable L2 without FP32 square overflow or a device FP64 requirement.

    Finite elements are scaled by their maximum before squaring; bounded FP32
    chunk sums are accumulated with Python FP64 math.fsum. Extreme scalar
    divisors use CPU FP64 scaling after transfer, avoiding reciprocal overflow
    or underflow on the device. Nonfinite elements
    are excluded ONLY from diagnostic finite_l2, never from a training tensor.
    """
    value = tensor.detach()
    if value.is_sparse:
        value = value.coalesce().values()
    if value.is_complex():
        value = value.abs()
    value = value.reshape(-1)
    count = value.numel()
    if count == 0:
        return dict(count=0, nonfinite_count=0, all_finite=True, abs_max=0.0, finite_abs_max=0.0, l2=0.0, finite_l2=0.0)
    maxima, finite_counts = [], []
    for chunk in value.split(chunk_elements):
        chunk = chunk.float()
        finite = torch.isfinite(chunk)
        finite_counts.append(int(finite.sum().item()))
        maxima.append(float(torch.where(finite, chunk.abs(), 0.0).max().item()))
    maximum = max(maxima, default=0.0)
    nonfinite = count - sum(finite_counts)
    parts = []
    if maximum:
        tiny = torch.finfo(torch.float32).tiny
        host_scale = maximum < tiny or maximum > 1.0 / tiny
        for chunk in value.split(chunk_elements):
            # A tiny FP32 scalar divisor can overflow its reciprocal; a huge
            # divisor can underflow it. Transfer first, THEN convert on CPU so
            # this exceptional path requires no PPU FP64 support.
            work = chunk.cpu().double() if host_scale else chunk.float()
            scaled = torch.where(torch.isfinite(work), work / maximum, 0.0)
            dtype = torch.float64 if host_scale else torch.float32
            parts.append(float(scaled.square().sum(dtype=dtype).item()))
    finite_l2 = maximum * math.sqrt(math.fsum(parts))
    return dict(count=count, nonfinite_count=nonfinite, all_finite=nonfinite == 0,
                abs_max=maximum if nonfinite == 0 else float("inf"), finite_abs_max=maximum,
                l2=finite_l2 if nonfinite == 0 else float("inf"), finite_l2=finite_l2)


def aggregate_statistics(named: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    rows = {}
    norms, maxima, finite_maxima, count, bad = [], [], [], 0, 0
    for name, tensor in named:
        row = tensor_statistics(tensor)
        rows[name] = row
        norms.append(row["finite_l2"])
        maxima.append(row["abs_max"])
        finite_maxima.append(row["finite_abs_max"])
        count += row["count"]
        bad += row["nonfinite_count"]
    finite_l2 = math.hypot(*norms)
    return dict(all_finite=bad == 0, count=count, nonfinite_count=bad,
                abs_max=max(maxima, default=0.0), finite_abs_max=max(finite_maxima, default=0.0),
                l2=finite_l2 if bad == 0 else float("inf"), finite_l2=finite_l2,
                tensors=rows)


GRADIENT_GROUPS = (
    "dit_backbone", "dit_input_projection", "dit_output_projection",
    "vae_encoder", "vae_decoder", "visual_adapter", "ray_embedding", "feature_norm", "gs_dit_kv", "gs_dit_ca", "gs_dit_layerscale", "gs_shared_kv", "gs_shared_ca", "gs_completion_gate", "gs_base", "other")

def _gradient_group(name: str) -> str:
    parts = [p for p in name.split(".") if p not in ("module", "_fsdp_wrapped_module")]
    name = ".".join(parts)
    if name.startswith("gs.enc_dec_backbone."):
        if name.endswith(".completion_gate"):
            return "gs_completion_gate"
        if name.startswith(("gs.enc_dec_backbone.kv_proj.", "gs.enc_dec_backbone.k_proj_norm.")):
            return "gs_shared_kv"
        if ".gs_cross_attn." in name or ".gs_cross_attn_scale." in name:
            return "gs_shared_ca"
        if ".dit_cross_attn_scale." in name:
            return "gs_dit_layerscale"
        if ".dit_cross_attn." in name:
            return "gs_dit_ca"
        if name.startswith(("gs.enc_dec_backbone.dit_kv_proj.",
                            "gs.enc_dec_backbone.dit_k_proj_norm.")):
            return "gs_dit_kv"
    if name.startswith("dit."):
        if ".patch_embedding." in name: return "dit_input_projection"
        if ".proj_out." in name: return "dit_output_projection"
        return "dit_backbone"
    for prefix, group in (("vae.vae.decoder.", "vae_decoder"), ("vae.vae.post_quant_conv.", "vae_decoder"),
                          ("vae.", "vae_encoder"), ("adapter.visual_patch.", "visual_adapter"),
                          ("adapter.ray_patch.", "ray_embedding"), ("adapter.output_norm.", "feature_norm"),
                          ("feature_norm.", "feature_norm"), ("gs.", "gs_base")):
        if name.startswith(prefix): return group
    return "other"


def finite_tensor_summary(named: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    by_device = {}
    total_elements = 0
    for name, tensor in named:
        value = tensor.detach()
        if value.is_sparse:
            value = value.coalesce().values()
        total_elements += value.numel()
        by_device.setdefault(value.device, []).append((name, torch.isfinite(value).all()))
    bad = []
    count = 0
    for entries in by_device.values():
        # One host transfer per device, rather than one synchronization per tensor.
        flags = torch.stack([value for _, value in entries]).cpu().tolist()
        count += len(entries)
        bad.extend(name for (name, _), finite in zip(entries, flags) if not finite)
    return {"all_finite": not bad, "tensor_count": count, "element_count": total_elements,
            "nonfinite_tensor_count": len(bad), "nonfinite_tensor_names": bad}


def expected_component_routes(component: str) -> dict[str, Any]:
    shared = ["dit_backbone", "dit_input_projection", "dit_output_projection"]
    expected = {"fm": shared, "gs": shared + ["visual_adapter", "ray_embedding", "feature_norm", "gs_base"]}
    return {"candidate_gradient_routes": expected.get(component, []),
            "always_frozen_target": "vae",
            "interpretation": "FM and GS reach the native velocity head; GS traverses frozen VAE decoder input derivatives",
            "zero_init_caveat": "new input columns are zero initially but can receive gradients"}


def capture_rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    # The trainer initializes the device before this function is invoked.
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        state["numpy"] = None
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        import numpy as np
        np.random.set_state(state["numpy"])


class NumericalMonitor:
    """Monitor a complete FSDP shard partition or replicated DDP parameters.

    sharded=True requires every parameter to belong to the FSDP shard partition.
    Do not pass a model mixing sharded and replicated trainable parameters.
    Per-tensor rows are rank-local shards; only global_* fields are reduced.
    """

    def __init__(self, model: torch.nn.Module, workspace: str | Path, rank: int = 0,
                 world_size: int = 1, sharded: bool = False, history_size: int = 100,
                 spike_factor: float = 100.0, detail_every: int = 100):
        self.model = model
        self.workspace = require_pfs(workspace)
        self.directory = self.workspace / "numerics"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank, self.world_size, self.sharded = rank, world_size, sharded
        self.spike_factor = spike_factor
        self.detail_every = max(1, int(detail_every))
        self.history = deque(maxlen=history_size)
        self.context: dict[str, Any] = {}
        self.first_anomaly = None
        self.activation_rows: dict[str, Any] = {}
        self.pre_reduce_rows: dict[str, Any] = {}
        self._pre_reduce_handles = []
        self._snapshot = None
        self._last_gradient_report = None
        self._history_recorded = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _reduce_scalar(self, value: float, operation: str) -> float:
        scalar = torch.tensor(value, device=self.device, dtype=torch.float32)
        if _distributed():
            dist.all_reduce(scalar, op=dist.ReduceOp.MAX if operation == "max" else dist.ReduceOp.SUM)
        return float(scalar.item())

    def coordinated_finite(self, local_finite: bool) -> bool:
        return self._reduce_scalar(0.0 if local_finite else 1.0, "max") == 0.0

    def _global_statistics(self, local: Mapping[str, Any]) -> dict[str, Any]:
        all_finite = self.coordinated_finite(bool(local["all_finite"]))
        maximum = self._reduce_scalar(float(local["abs_max"]), "max")
        local_l2 = float(local["finite_l2"])
        scale = self._reduce_scalar(float(local["finite_abs_max"]), "max")
        # Scale by an element magnitude, not a local L2 that could exceed FP32.
        # The squared ratios sum to at most the global parameter element count.
        if scale:
            sum_scaled = self._reduce_scalar((local_l2 / scale) ** 2, "sum")
            copies = 1 if self.sharded else (_actual_world_size())
            finite_l2 = scale * math.sqrt(sum_scaled / copies)
        else:
            finite_l2 = 0.0
        all_finite = all_finite and math.isfinite(finite_l2)
        return dict(all_finite=all_finite, global_finite_abs_max=scale,
                    global_l2=finite_l2 if all_finite else float("inf"),
                    global_finite_l2=finite_l2, global_abs_max=maximum,
                    global_norm_semantics="sum_unique_fsdp_shards" if self.sharded else "mean_replicated_rank_squared_norms")

    def begin_step(self, step: int, batch: Mapping[str, Any], seed: int | None = None) -> None:
        self.context = {"step": int(step), "rank": self.rank, "world_size": self.world_size,
                        "seed": seed, "time_unix": time.time()}
        for key in ("metadata_json", "frame_ids", "native_frame_ids", "valid_view_mask", "observed_mask", "case_id", "provenance"):
            if key in batch:
                self.context[key] = _json_value(batch[key])
        self.first_anomaly = None
        self.activation_rows = {}
        self.pre_reduce_rows = {}
        self._last_gradient_report = None
        self._history_recorded = False

    def record_activation(self, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        row = tensor_statistics(tensor)
        previous = self.activation_rows.get(name)
        self.activation_rows[name] = {
            **row, "calls": 1 if previous is None else previous["calls"] + 1,
            "all_calls_finite": row["all_finite"] and
                                (previous is None or previous["all_calls_finite"]),
            "max_call_l2": max(row["l2"], 0.0 if previous is None else previous["max_call_l2"]),
            "max_call_abs": max(row["abs_max"], 0.0 if previous is None else previous["max_call_abs"])}
        if not row["all_finite"] and self.first_anomaly is None:
            self.first_anomaly = {"kind": "activation", "name": name, **row}

    def attach_pre_reduce_hooks(self, raw_model: torch.nn.Module) -> None:
        """Call before FSDP construction. Never throw from a backward hook."""
        if self._pre_reduce_handles:
            raise RuntimeError("Pre-reduce hooks already attached")
        for name, parameter in raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            def hook(gradient, parameter_name=name):
                row = tensor_statistics(gradient)
                # A shared parameter may receive several hook calls on recomputation.
                self.pre_reduce_rows.setdefault(parameter_name, []).append(row)
                if not row["all_finite"] and self.first_anomaly is None:
                    self.first_anomaly = {"kind": "local_pre_reduce_gradient",
                                          "name": parameter_name, **row}
                return gradient
            self._pre_reduce_handles.append(parameter.register_hook(hook))

    def close(self) -> None:
        for handle in self._pre_reduce_handles:
            handle.remove()
        self._pre_reduce_handles.clear()

    def inspect_gradients(self, stage: str = "post_reduce_pre_clip",
                          update_history: bool = True) -> dict[str, Any]:
        local = aggregate_statistics(
            (name, parameter.grad) for name, parameter in self.model.named_parameters()
            if parameter.grad is not None)
        report = {**self.context, "stage": stage,
                  "tensor_rows_scope": "rank_local_fsdp_shard" if self.sharded else "rank_replica",
                  "local": local, **self._global_statistics(local)}
        activation_finite = self.coordinated_finite(self.first_anomaly is None)
        report["all_finite"] = report["all_finite"] and activation_finite
        report["activation_or_local_gradient_finite"] = activation_finite
        report["first_local_anomaly"] = self.first_anomaly
        report["activations"] = self.activation_rows
        report["local_pre_reduce"] = self.pre_reduce_rows if self._pre_reduce_handles else None
        report["local_pre_reduce_semantics"] = (
            "per_hook_unreduced_contributions_not_global_norm" if self._pre_reduce_handles
            else "unavailable_hooks_not_installed_before_fsdp")
        baseline = statistics.median(self.history) if len(self.history) >= 5 else None
        report["healthy_history_median"] = baseline
        report["spike"] = bool(report["all_finite"] and baseline is not None
                               and report["global_l2"] > max(baseline, 1e-30) * self.spike_factor)
        report["spike_threshold_semantics"] = "capture_only_not_automatic_rejection"
        if update_history and not self._history_recorded and report["all_finite"] and not report["spike"]:
            self.history.append(report["global_l2"])
            self._history_recorded = True
        self._last_gradient_report = report
        report["module_gradient_coverage"] = self.gradient_coverage(local["tensors"])
        detailed = (int(self.context.get("step", 0)) % self.detail_every == 0 or
                    report["spike"] or not report["all_finite"] or stage.startswith("component_"))
        log_report = report if detailed else {
            **report, "local": {key: value for key, value in local.items() if key != "tensors"},
            "activations": {"detailed_rows_omitted": True,
                            "recorded_sites": len(self.activation_rows)},
            "local_pre_reduce": {"detailed_rows_omitted": True,
                                 "recorded_parameters": len(self.pre_reduce_rows)}}
        append_jsonl(self.directory / f"gradients_rank{self.rank}.jsonl", log_report)
        return report

    @torch.no_grad()
    def clip_global(self, max_norm: float = 1.0) -> dict[str, Any]:
        before = self._last_gradient_report or self.inspect_gradients()
        if not before["all_finite"]:
            return {**before, "clip_scale": None, "update_permitted": False}
        scale = min(1.0, float(max_norm) / (before["global_l2"] + 1e-12))
        element_scale = before["global_finite_abs_max"]
        for parameter in self.model.parameters():
            if parameter.grad is not None and scale < 1.0:
                # Two bounded operations avoid casting a subnormal clip factor
                # to BF16 or casting a >FP32 global L2 to a device scalar.
                work = parameter.grad.float() / element_scale
                work.mul_(float(max_norm) * (element_scale / before["global_l2"]))
                parameter.grad.copy_(work)
        after = self.inspect_gradients("post_clip", update_history=False)
        after.update(clip_scale=scale, pre_clip_global_l2=before["global_l2"],
                     update_permitted=after["all_finite"])
        append_jsonl(self.directory / f"clipping_rank{self.rank}.jsonl", {
            **self.context, "all_finite": after["all_finite"], "clip_scale": scale,
            "pre_clip_global_l2": before["global_l2"], "post_clip_global_l2": after["global_l2"]})
        return after

    def gradient_coverage(self, rows: Mapping[str, Any]) -> dict[str, Any]:
        groups = {name: {"parameter_tensors": 0, "parameter_elements": 0,
                         "gradient_present_tensors": 0, "gradient_nonzero_tensors": 0,
                         "gradient_nonfinite_tensors": 0, "gradient_elements": 0}
                  for name in GRADIENT_GROUPS}
        for name, parameter in self.model.named_parameters():
            group = groups[_gradient_group(name)]
            if parameter.numel() == 0:
                continue
            group["parameter_tensors"] += 1
            group["parameter_elements"] += parameter.numel()
            if name in rows:
                row = rows[name]
                group["gradient_present_tensors"] += 1
                group["gradient_elements"] += row["count"]
                group["gradient_nonzero_tensors"] += int(row["finite_l2"] > 0)
                group["gradient_nonfinite_tensors"] += int(not row["all_finite"])
        flags = torch.tensor(
            [[int(group["gradient_present_tensors"] > 0),
              int(group["gradient_nonzero_tensors"] > 0),
              int(group["gradient_nonfinite_tensors"] > 0)] for group in groups.values()],
            device=self.device, dtype=torch.int32)
        if _distributed():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        for group, values in zip(groups.values(), flags.cpu().tolist()):
            group.update(global_any_gradient=bool(values[0]),
                         global_any_nonzero_gradient=bool(values[1]),
                         global_any_nonfinite_gradient=bool(values[2]),
                         count_scope="rank_local_shard" if self.sharded else "rank_replica")
        return groups

    def capture_optimizer_state(self, optimizer: torch.optim.Optimizer,
                                ema: Mapping[str, Any] | None = None,
                                detailed: bool = False) -> dict[str, Any]:
        named_parameters = dict(self.model.named_parameters())
        moment_tensors, steps = [], {}
        names_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
        for parameter, state in optimizer.state.items():
            name = names_by_id.get(id(parameter), f"optimizer_parameter_{len(steps)}")
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    moment_tensors.append((f"{name}/{key}", state[key]))
            if "step" in state:
                if torch.is_tensor(state["step"]):
                    moment_tensors.append((f"{name}/step", state["step"]))
                elif not math.isfinite(float(state["step"])):
                    moment_tensors.append((f"{name}/step", torch.tensor(float(state["step"]))))
                if detailed:
                    steps[name] = float(state["step"].item()) if torch.is_tensor(state["step"]) else state["step"]
        sections = {}
        for name, tensors in (("parameters", named_parameters.items()),
                              ("adam_moments", moment_tensors), ("ema", _named_tensors(ema or {}))):
            tensors = list(tensors)
            if detailed:
                local = aggregate_statistics(tensors)
                sections[name] = {"local": local, **self._global_statistics(local)}
            else:
                local = finite_tensor_summary(tensors)
                finite = self.coordinated_finite(local["all_finite"])
                sections[name] = {"local": local, "all_finite": finite,
                                  "norms": "omitted_between_capture_steps"}
        result = {**self.context, **sections, "detailed": detailed,
                  "optimizer_steps": steps if detailed else "recorded_at_capture_steps",
                  "all_finite": all(section["all_finite"] for section in sections.values())}
        append_jsonl(self.directory / f"state_rank{self.rank}.jsonl", result)
        return result



    def save_failure_context(self, batch: Mapping[str, Any], reason: str,
                             extra: Mapping[str, Any] | None = None) -> str:
        path = self.directory / f"failure_step{self.context.get('step', -1)}_rank{self.rank}.pt"
        state = {"context": self.context, "reason": reason,
                 "first_anomaly": self.first_anomaly, "rng": capture_rng_state(),
                 "batch": _cpu_clone(batch), "extra": _cpu_clone(extra or {})}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return str(path)


def _actual_world_size() -> int:
    return dist.get_world_size() if _distributed() else 1


def _cpu_clone(value: Any) -> Any:
    if hasattr(value, "local_shards"):
        return {"local_shards": [
            {"tensor": shard.tensor.detach().cpu().clone(),
             "offsets": list(shard.metadata.shard_offsets),
             "sizes": list(shard.metadata.shard_sizes)}
            for shard in value.local_shards()]}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value
