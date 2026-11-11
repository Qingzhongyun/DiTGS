"""Native shared-CA initialization and zero-initialized learned gates."""
import torch
from safetensors import safe_open

def load_base_gs(core,ray_patch,path,adapter_output_norm=None):
    """Exact official native-key load; only new learned gates initialize to zero."""
    all_state=dict(core.state_dict())
    gates={k:v for k,v in all_state.items() if k.endswith(".completion_gate")}
    native={k:v for k,v in all_state.items() if k not in gates}
    if len(core.enc_dec_backbone.decoder_blocks)!=12 or len(gates)!=(12 if core.attention_variant=="gated" else 0):
        raise ValueError("Historical attention topology/gate count mismatch")
    with safe_open(path,framework="pt",device="cpu") as handle:
        keys=set(handle.keys())
        if keys!=set(native):
            raise ValueError(f"Official native GS keys differ missing={set(native)-keys},unexpected={keys-set(native)}")
        with torch.no_grad():
            for key,destination in native.items():
                value=handle.get_tensor(key)
                if value.shape!=destination.shape:
                    raise ValueError("Official GS tensor shape mismatch: "+key)
                destination.copy_(value)
            for destination in gates.values():
                destination.zero_()
            for key,destination in ray_patch.state_dict().items():
                value=handle.get_tensor("patch_plucker_embed."+key)
                if value.shape!=destination.shape:
                    raise ValueError("Adapter ray patch shape mismatch: "+key)
                destination.copy_(value)
            if adapter_output_norm is not None:
                adapter_output_norm.load_state_dict(core.enc_dec_backbone.encoder_norm.state_dict(),strict=True)
