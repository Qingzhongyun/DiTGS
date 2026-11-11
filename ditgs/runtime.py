"""Distributed storage observability."""
import torch

def fsdp_storage_summary(model):
    """Read allocator storage metadata only; never gather/copy managed tensors.

    PyTorch2.8 FSDP1 private storage names are used only for diagnostics.
    Missing fields are explicit; this routine never changes FSDP state.
    """
    full_bytes = grad_bytes = 0
    units = retained_units = unknown_units = 0
    for module in model.modules():
        handle = getattr(module, "_handle", None)
        if handle is None or not hasattr(handle, "flat_param"):
            continue
        units += 1
        flat = handle.flat_param
        full = getattr(flat, "_full_param_padded", None)
        if full is None:
            unknown_units += 1
        else:
            size = full.untyped_storage().nbytes()
            full_bytes += size
            retained_units += int(size > 0)
        if flat.grad is not None:
            grad_bytes += flat.grad.numel() * flat.grad.element_size()
    return {"fsdp_units": units, "units_retaining_full_parameter_storage": retained_units,
            "full_parameter_storage_bytes": full_bytes, "flat_gradient_view_bytes": grad_bytes,
            "units_without_full_storage_metadata": unknown_units}
