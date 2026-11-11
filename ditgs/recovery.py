"""Conservative training rejection policy; never changes the benchmark cohort."""
import math
import statistics

def loss_reasons(losses, cfg, history=()):
    reasons = []
    for name, value in losses.items():
        if not math.isfinite(value) or abs(value) > cfg.loss_abs_max:
            reasons.append("loss_range:" + name)
    fm = losses.get("fm")
    if fm is not None and fm < 1e-12:
        reasons.append("implausibly_small_fm")
    total = sum(getattr(cfg, name+"_weight") * losses.get(name, 0.) for name in ("fm", "img", "gs"))
    if len(history) >= 20 and total > max(statistics.median(history), 1e-12) * cfg.diagnostic_spike_ratio:
        reasons.append("loss_history_spike")
    return reasons

def gs_gradient_groups(cfg):
    if cfg.variant=="residual":
        groups=("visual_adapter","ray_embedding","feature_norm","gs_dit_kv","gs_dit_ca","gs_dit_layerscale")
    else:
        groups=("visual_adapter","ray_embedding","feature_norm","gs_shared_kv","gs_shared_ca")
        if cfg.variant=="gated":groups+=("gs_completion_gate",)
    return groups if cfg.freeze_original_gs else groups+("gs_base",)


def gradient_reasons(report, cfg):
    reasons = []
    norm = report["global_l2"]
    if not report["all_finite"] or not math.isfinite(norm):
        reasons.append("nonfinite_gradient_or_activation")
    elif norm < cfg.gradient_norm_min or norm > cfg.gradient_norm_max:
        reasons.append("gradient_norm_out_of_range")
    if report["spike"]:
        reasons.append("gradient_history_spike")
    required = ["dit_input_projection"] if cfg.train_dit_condition else []
    if not cfg.freeze_dit:
        required += ["dit_backbone", "dit_output_projection"]
    if cfg.use_gs:
        required += list(gs_gradient_groups(cfg))
    coverage = report["module_gradient_coverage"]
    for name in required:
        group = coverage[name]
        if name in gs_gradient_groups(cfg):
            # Zero contribution can occur after local point filtering. Missing graph is different.
            present = group.get("global_any_gradient", group["global_any_nonzero_gradient"])
        else:
            present = group["global_any_nonzero_gradient"]
        if not present:
            reasons.append("missing_gradient:" + name)
    return reasons


def resume_decision(cfg, workspace, *, explicit=False, restarting=False):
    """Honor manual exact resume; only recovery resumes advance the failure cursor."""
    import json
    from pathlib import Path
    pointer = Path(workspace) / "latest_safe.json"
    if explicit and not restarting and cfg.resume:
        return {"checkpoint": cfg.resume, "use_failure_cursor": False}
    checkpoint = json.loads(pointer.read_text())["checkpoint"] if pointer.is_file() else cfg.resume
    return {"checkpoint": checkpoint, "use_failure_cursor": True}
