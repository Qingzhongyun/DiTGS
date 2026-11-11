"""Separate optimizer, any-GS, all-source, image-VJP and visibility update counters."""
def update_health(progress, record, use_gs):
    accepted = bool(record.get("accepted",False))
    progress["attempted_updates"] = progress.get("attempted_updates",0)+1
    progress["rejected_updates"] = progress.get("rejected_updates",0)+int(not accepted)
    progress["consecutive_rejections"] = 0 if accepted else progress.get("consecutive_rejections",0)+1
    flags = {
        "gs": accepted and use_gs and bool(record.get("gs_has_nonzero_gradient",False)),
        "all_source_gs": accepted and use_gs and bool(record.get("gs_all_sources_nonzero",False)),
        "renderer": accepted and use_gs and bool(record.get("renderer_has_nonzero_gradient",False)),
        "visibility": accepted and use_gs and bool(record.get("visibility_has_nonzero_gradient",False))}
    for name,good in flags.items():
        key = "gs_gradient_updates" if name=="gs" else name+"_gradient_updates"
        progress[key] = progress.get(key,0)+int(good)
        if name in ("gs","renderer"):
            key = "consecutive_without_"+name+"_update"
            progress[key] = 0 if good else progress.get(key,0)+int(use_gs)
    state = ("no_effective_updates" if progress["consecutive_rejections"]>=32 else
             "no_gs_gradient_updates" if progress["consecutive_without_gs_update"]>=32 else
             "no_image_renderer_updates" if progress["consecutive_without_renderer_update"]>=32 else "training")
    return {"state":state,"attempted_updates":progress["attempted_updates"],
        "accepted_updates":progress["accepted_updates"],"rejected_updates":progress["rejected_updates"],
        "acceptance_rate":(progress["attempted_updates"]-progress["rejected_updates"])/max(progress["attempted_updates"],1),
        **{k:v for k,v in progress.items() if k.endswith("gradient_updates") or k.startswith("consecutive_")},
        "gs_update_this_step":flags["gs"],"renderer_update_this_step":flags["renderer"],
        "visibility_update_this_step":flags["visibility"],"all_sources_update_this_step":flags["all_source_gs"]}


def collapse_reason(progress, cfg):
    if progress.get("consecutive_without_gs_update",0) >= cfg.max_without_gs_update:
        return "sustained_no_trainable_GS_parameter_gradient"
    if progress.get("consecutive_without_renderer_update",0) >= cfg.max_without_renderer_update:
        return "sustained_no_image_renderer_gradient"
    return None
