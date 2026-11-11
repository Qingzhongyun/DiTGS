"""Low-synchronization observability; training/safety thresholds stay unchanged.

Only compact device reductions cross the host boundary on ordinary steps.
Full reference statistics remain available at scheduled steps or global anomalies.
"""
from __future__ import annotations
import json
import math
import statistics
import time
from pathlib import Path
import torch
import torch.distributed as dist
from .numerics import (
    NumericalMonitor as ReferenceMonitor, GRADIENT_GROUPS, _gradient_group,
    _json_value, _distributed, _actual_world_size)

class BufferedJsonl:
    """Normal diagnostics may lose at most the unflushed tail on a hard kill.
    Safety cursors/checkpoint manifests use the original atomic writer, not this.
    """
    def __init__(self):
        self.handles = {}
        self.pending = 0
        self.last_flush = time.monotonic()

    def append(self, path, payload, *, force=False):
        path = Path(path)
        if path not in self.handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handles[path] = path.open("a", encoding="utf-8")
        self.handles[path].write(json.dumps(_json_value(payload), ensure_ascii=False)+"\n")
        self.pending += 1
        if force or self.pending >= 10 or time.monotonic()-self.last_flush >= 30:
            self.flush()

    def flush(self):
        for handle in self.handles.values():
            handle.flush()
        self.pending = 0
        self.last_flush = time.monotonic()

    def close(self):
        self.flush()
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

class NumericalMonitor(ReferenceMonitor):
    def __init__(self, *args, writer, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        # Cache parameter objects/names, never transient .grad or FSDP views.
        self._named = tuple(self.model.named_parameters())
        self._groups = {name: _gradient_group(name) for name, _ in self._named}
        self._activation_flags = {}

    def begin_step(self, step, batch, seed=None):
        super().begin_step(step, batch, seed)
        self._activation_flags = {}

    def _detail_due(self):
        return int(self.context.get("step", 0)) % self.detail_every == 0

    @torch.no_grad()
    def record_activation(self, name, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        finite = torch.isfinite(tensor.detach()).all()
        previous = self._activation_flags.get(name)
        self._activation_flags[name] = finite if previous is None else previous & finite
        if self._detail_due():
            super().record_activation(name, tensor)

    def _identify_activation_failure(self):
        if not self._activation_flags:
            return
        names, flags = zip(*self._activation_flags.items())
        for name, valid in zip(names, torch.stack(flags).cpu().tolist()):
            if not valid and self.first_anomaly is None:
                self.first_anomaly = {"kind": "activation", "name": name, "all_finite": False}

    def _emit(self, filename, payload):
        if self._detail_due() or not payload.get("all_finite", True):
            self.writer.append(self.directory/filename, payload,
                               force=not payload.get("all_finite", True))

    @torch.no_grad()
    def inspect_gradients(self, stage="post_reduce_pre_clip", update_history=True):
        if self._detail_due():
            return super().inspect_gradients(stage, update_history)
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        maxima, bads, gradients = [], [], []
        grouped = {name: [] for name in GRADIENT_GROUPS}
        elements = 0
        for name, parameter in self._named:
            grad = parameter.grad
            if grad is None or grad.numel() == 0:
                continue
            value = grad.detach()
            # Storage is FP32. Do not retain full-size scratch tensors.
            finite = torch.isfinite(value)
            maximum = torch.where(finite, value.abs(), 0.0).amax().float()
            bad = (~finite).any().float()
            maxima.append(maximum)
            bads.append(bad)
            grouped[self._groups[name]].append((maximum, bad))
            gradients.append(value)
            elements += value.numel()
        local_max = torch.stack(maxima).amax() if maxima else zero
        local_bad = torch.stack(bads).amax() if bads else zero
        activation_bad = ((~torch.stack(list(self._activation_flags.values()))).any().float()
                          if self._activation_flags else zero)
        header = [local_max, local_bad, activation_bad]
        for name in GRADIENT_GROUPS:
            rows = grouped[name]
            header.extend([
                zero.new_tensor(float(bool(rows))),
                torch.stack([maximum > 0 for maximum, _ in rows]).any().float() if rows else zero,
                torch.stack([bad for _, bad in rows]).amax() if rows else zero])
        header = torch.stack(header)
        if _distributed():
            dist.all_reduce(header, op=dist.ReduceOp.MAX)
        packed = header.cpu().tolist()  # One compact host transfer, not one per tensor.
        maximum = packed[0]
        tiny = torch.finfo(torch.float32).tiny
        extreme = maximum > 0 and (maximum < tiny or maximum > 1.0/tiny)
        # This branch is based entirely on globally reduced flags: all ranks
        # enter the reference collectives together, including healthy ranks.
        if packed[1] or packed[2] or extreme:
            self._identify_activation_failure()
            return super().inspect_gradients(stage, update_history)
        divisor = header[0] if maximum else header.new_tensor(1.0)
        sums = []
        for value in gradients:
            for chunk in value.reshape(-1).split(1_048_576):
                sums.append((chunk.float()/divisor).square().sum(dtype=torch.float32))
        square_sum = torch.stack(sums).sum() if sums else zero.clone()
        if _distributed():
            dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
        copies = 1 if self.sharded else _actual_world_size()
        norm = maximum * math.sqrt(float(square_sum.item())/copies)
        coverage = {}
        for index, name in enumerate(GRADIENT_GROUPS):
            present, nonzero, bad = packed[3+3*index:6+3*index]
            coverage[name] = {
                "global_any_gradient": bool(present),
                "global_any_nonzero_gradient": bool(nonzero),
                "global_any_nonfinite_gradient": bool(bad),
                "count_scope": "global_flags_only"}
        baseline = statistics.median(self.history) if len(self.history) >= 5 else None
        spike = bool(math.isfinite(norm) and baseline is not None and
                     norm > max(baseline, 1e-30)*self.spike_factor)
        report = {
            **self.context, "stage": stage, "all_finite": math.isfinite(norm),
            "global_l2": norm, "global_finite_l2": norm,
            "global_abs_max": maximum, "global_finite_abs_max": maximum,
            "module_gradient_coverage": coverage,
            "healthy_history_median": baseline, "spike": spike,
            "first_local_anomaly": None,
            "local": {"count": elements, "tensor_count": len(gradients)},
            "gradient_statistics": "device_scaled_sum_global_shards"}
        # Rare spikes still retain the original detailed diagnosis.
        if spike or not report["all_finite"]:
            return super().inspect_gradients(stage, update_history)
        if update_history and not self._history_recorded:
            self.history.append(norm)
            self._history_recorded = True
        self._last_gradient_report = report
        return report

    @torch.no_grad()
    def clip_global(self, max_norm=1.0):
        before = self._last_gradient_report
        if before is None:
            before = self.inspect_gradients(update_history=False)
        if not before["all_finite"]:
            return {**before, "clip_scale": None, "update_permitted": False}
        norm, maximum = before["global_l2"], before["global_finite_abs_max"]
        scale = min(1.0, float(max_norm)/(norm+1e-12))
        flags = []
        for _, parameter in self._named:
            if parameter.grad is None or parameter.grad.numel() == 0:
                continue
            if scale < 1.0:
                # Preserve the existing two-stage, bounded clipping arithmetic.
                work = parameter.grad.float()/maximum
                work.mul_(float(max_norm)*(maximum/norm))
                parameter.grad.copy_(work)
            flags.append(torch.isfinite(parameter.grad).all())
        valid = (torch.stack(flags).all().float() if flags
                 else torch.ones((), device=self.device))
        if _distributed():
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        permitted = bool(valid.item())
        result = {**before, "all_finite": permitted, "clip_scale": scale,
                  "update_permitted": permitted, "pre_clip_global_l2": norm,
                  "post_clip_l2_estimate": float(max_norm) if scale < 1.0 else norm,
                  "post_clip_norm_measured": False}
        self._emit(f"clipping_rank{self.rank}.jsonl", {
            **self.context, **{k: result[k] for k in (
                "all_finite", "clip_scale", "pre_clip_global_l2",
                "post_clip_l2_estimate", "post_clip_norm_measured")}})
        return result

    @torch.no_grad()
    def capture_optimizer_state(self, optimizer, ema=None, detailed=False):
        # Range comparisons reject NaN/Inf as well as finite-but-corrupt values.
        # This replaces BOTH the finite scan and the separate trainer range scan.
        checks = {"parameters": [], "adam_moments": [], "ema": []}
        def add(section, name, value, bound=None, nonnegative=False):
            value = value.detach()
            if value.numel() == 0:
                return
            if bound is None:
                ok = torch.isfinite(value).all()
            elif nonnegative:
                ok = ((value >= 0) & (value <= bound)).all()
            else:
                ok = (value.abs() <= bound).all()
            checks[section].append((name, ok))
        for name, parameter in self._named:
            add("parameters", name, parameter, 1e6)
        for name, value in (ema or {}).items():
            add("ema", name, value, 1e6)
        names = {id(parameter): name for name, parameter in self._named}
        for parameter, state in optimizer.state.items():
            name = names.get(id(parameter), "optimizer_parameter")
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"):
                if key not in state:
                    continue
                value = state[key]
                if not torch.is_tensor(value):
                    value = torch.tensor(float(value))
                # AMSGrad is disabled; preserve finite checking if ever present.
                bound = 1000 if key in ("exp_avg", "exp_avg_sq") else None
                add("adam_moments", name+"/"+key, value, bound, key == "exp_avg_sq")
        local = []
        for rows in checks.values():
            gpu = [ok for _, ok in rows if ok.device.type != "cpu"]
            cpu = [ok for _, ok in rows if ok.device.type == "cpu"]
            cpu_ok = bool(torch.stack(cpu).all()) if cpu else True
            flag = (torch.stack(gpu).all() if gpu
                    else torch.ones((), device=self.device, dtype=torch.bool))
            local.append(flag & cpu_ok)
        flags = torch.stack(local).float()
        if _distributed():
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        valid = flags.cpu().tolist()
        result = {**self.context, "all_finite": all(valid), "detailed": detailed,
                  "validation": "finite_and_existing_range_checks_combined"}
        for section, value in zip(checks, valid):
            result[section] = {"all_finite": bool(value)}
        if detailed or not result["all_finite"]:
            # All ranks agree on this branch; reference diagnostics are not a
            # prerequisite for the ordinary-step safety verdict.
            detail = super().capture_optimizer_state(optimizer, ema, detailed=True)
            result["reference_statistics"] = detail
            result["all_finite"] = result["all_finite"] and detail["all_finite"]
            self.writer.append(self.directory/f"state_bounds_rank{self.rank}.jsonl",
                               result, force=not result["all_finite"])
        return result
