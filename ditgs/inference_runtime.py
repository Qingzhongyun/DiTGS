"""Inference-only1/2-view dispatch; trained parameter/buffer names stay unchanged."""
from pathlib import Path
import zipfile
import numpy as np
import torch
from einops import rearrange
from .residual_attention import SequentialSourceBackbone,gather_views
from .geometry import tokengs_plucker_maps,flashworld_camera_raymaps,validate_cameras
from .sampling_gs import finite_or_raise
from .gaussian_safety import prepare_gaussians
from .model import MultiViewModel
from .eval_data import BenchmarkData
from .eval_protocol import seed_for,zip_layout
from tokengs.data.camera_preprocessing import rectified_intrinsics

class VariableViewSequentialSourceBackbone(SequentialSourceBackbone):
    @torch.no_grad()
    def sparse_image_features(self, images, plucker, mask):
        count, total = self.eval_input_views, self.eval_total_views
        if (mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[1] != total
                or not bool((mask.sum(1) == count).all())):
            raise ValueError('Require configured [B,T] mask with N real slots')
        b = mask.shape[0]
        if (plucker.ndim != 5 or plucker.shape[:3] != (b, total, 6)
                or images.ndim != 5 or images.shape[0] != b or images.shape[2] != 3
                or images.shape[-2:] != plucker.shape[-2:]):
            raise ValueError('Selected real RGB and calibrated rays must share the same raster')
        ids = torch.arange(total, device=mask.device).expand_as(mask).masked_select(mask).reshape(b, count)
        if images.shape[1] == total:
            real_rgb = gather_views(images, ids)
        elif images.shape[1] == count:
            real_rgb = images  # Compact RGB in ascending observed-slot order.
        else:
            raise ValueError('Invalid real image count')
        real_rays = gather_views(plucker, ids)
        mean = real_rgb.new_tensor((.485,.456,.406)).reshape(1,1,3,1,1)
        std = real_rgb.new_tensor((.229,.224,.225)).reshape(1,1,3,1,1)
        embedded = self.patch_embed(((real_rgb-mean)/std).flatten(0,1))
        embedded = embedded+self.patch_plucker_embed(real_rays.flatten(0,1).to(embedded))
        return self.enc_dec_backbone._encode_features(
            embedded.reshape(b, count*embedded.shape[1], embedded.shape[2]))

    def source_keys_values(self, gs_features, dit_features):
        count, total = self.eval_input_views, self.eval_total_views
        if gs_features.ndim != 3 or dit_features.ndim != 3:
            raise ValueError('Require independent [B,N*P,C] and [B,T*P,C] memories')
        b, real_tokens, channels = gs_features.shape
        if (real_tokens <= 0 or real_tokens % count or channels != self.opt.enc_embed_dim
                or dit_features.shape != (b,total*(real_tokens//count),channels)):
            raise ValueError('Variable real/DiT source memory grids differ')
        core = self.enc_dec_backbone
        gs_keys, gs_values = core._features_to_kv(gs_features)
        dit_keys, dit_values = rearrange(core.dit_kv_proj(dit_features),
            'b n (kv h c) -> kv b h n c', kv=2, h=self.opt.enc_num_heads)
        dit_keys = core.dit_k_proj_norm(dit_keys)
        return gs_keys, gs_values, dit_keys, dit_values

class InferenceBackbone(VariableViewSequentialSourceBackbone):
    def source_keys_values(self,real,prior):
        if self.source_variant=="residual":return super().source_keys_values(real,prior)
        b,n,c=real.shape
        if n%self.eval_input_views or prior.shape!=(b,(self.eval_total_views-self.eval_input_views)*(n//self.eval_input_views),c):
            raise ValueError("Shared-CA inference source grids differ")
        keys,values=self.enc_dec_backbone._features_to_kv(torch.cat((real,prior),dim=1))
        return keys[:,:,:n],values[:,:,:n],keys[:,:,n:],values[:,:,n:]

class InferenceModel(MultiViewModel):
    def configure_views(self,count):
        if count not in (1,2):raise ValueError("Supported input counts:1or2")
        self.eval_input_views=count;self.eval_total_views=8
        self.gs.__class__=InferenceBackbone
        self.gs.eval_input_views=count;self.gs.eval_total_views=8;self.gs.source_variant=self.cfg.variant
        for block in self.gs.enc_dec_backbone.decoder_blocks:
            block.gs_cross_attn.observed_views=count

    def source_gs_memories(self,features,images,c2w,intrinsics,mask):
        rays=tokengs_plucker_maps(c2w,intrinsics,*images.shape[-2:])
        real=self.gs.sparse_image_features(images,rays,mask)
        prior=self.adapter(features,rays)
        if self.cfg.variant!="residual":
            b=mask.shape[0];p=prior.shape[1]//8
            ids=torch.arange(8,device=mask.device).expand_as(mask).masked_select(~mask).reshape(b,8-self.eval_input_views)
            prior=gather_views(prior.reshape(b,8,p,-1),ids).reshape(b,(8-self.eval_input_views)*p,-1)
        return real,prior

    def forward(self,operation,**values):
        if operation=="encode_context":
            return self.vae(values["images"],generator=values["generator"])
        if operation=="velocity":
            return self.dit(values["state"],values["sigma"],values["condition"],values["rays"],self.empty_text)
        if operation!="reconstruct_gs":raise ValueError(operation)
        check=values.get("check",finite_or_raise)
        features=self.vae(values["state"],operation="decode_features");check("vae_features",features)
        real,prior=self.source_gs_memories(features,values["observed_images"],values["c2w"],values["intrinsics"],values["observed_mask"])
        check("real_memory",real);check("prior_memory",prior)
        self.gs.activation_head.detailed_statistics=False
        points=self.gs.gaussians_from_source_features(real,prior,capture_stats=False)
        points,active,stats=prepare_gaussians(points,detailed=False)
        invalid=getattr(self.gs.activation_head,"local_invalid_mask",None)
        if invalid is not None:active=active & ~invalid
        check("gaussians",points)
        return points,active,stats

def rebase_dit_cameras(c2w):
    """Keep camera algebra FP32 even inside the BF16 model-inference context.

    Casting inputs with .float() alone does not protect matmul from autocast.
    Do not overwrite the first pose or weaken the downstream gauge check: compute
    the actual relative transforms accurately, leaving the GS camera tensor intact.
    """
    with torch.autocast(device_type=c2w.device.type, enabled=False):
        cameras = c2w.float()
        return torch.linalg.inv(cameras[:, 0])[:, None] @ cameras

@torch.no_grad()
def sample_latents(model,context,c2w,intrinsics,frames,context_slots,*,height,width,nfe,
                   seed,scene_id,scene_scale=.15,normalization_eps=.01,check=finite_or_raise):
    total,count = len(frames),len(context_slots)
    if type(nfe) is not int or not 1 <= nfe <= 256 or frames != sorted(set(frames)):
        raise ValueError('Invalid Euler budget or unique ordered camera slots')
    if (not 0 < count < total or context_slots != sorted(set(context_slots))
            or min(context_slots)<0 or max(context_slots)>=total
            or c2w.shape != (1,total,4,4) or intrinsics.shape != (1,total,4)):
        raise ValueError('Invalid variable camera/condition layout')
    b,c,n,h,w = context.shape
    if b != 1 or n != count or (height//16,width//16) != (h,w):
        raise ValueError('Selected context latent raster mismatch')
    condition = context.new_zeros(1,c,total,h,w)
    condition[:,:,context_slots] = context
    state = torch.cat([torch.randn((1,c,1,h,w),dtype=torch.float32,device=context.device,
        generator=torch.Generator(device=context.device).manual_seed(seed_for(seed,scene_id,f,'noise')))
        for f in frames],dim=2)
    # GS remains in first REAL-context gauge. Max-min extension may add a camera
    # before that context, so rebase ONLY DiT raymap poses to the first dense slot.
    # Keep scene_scale in these translations; FlashWorld divides by its radius+eps.
    dit_c2w = rebase_dit_cameras(c2w)
    rays = flashworld_camera_raymaps(dit_c2w,intrinsics,height,width,h,w,
        scene_scale=scene_scale,normalization_eps=normalization_eps)
    check('condition',condition);check('rays',rays)
    for i in range(nfe):
        velocity = model('velocity',state=state,sigma=state.new_full((1,),1-i/nfe),condition=condition,rays=rays)
        state = state-velocity.float()/nfe
        check(f'euler_{i}',torch.stack((state,velocity.float())))
    return state

class MultiviewBenchmarkData(BenchmarkData):
    def load_condition(self,row):
        from ditgs.archive import DL3DVArchive
        path = self.root/(row['scene_id']+'.zip')
        required = max(row['context']+row['target'])+1
        with zipfile.ZipFile(path) as z:
            prefix,meta = zip_layout(z,path)
            if len(meta['frames']) < required:
                raise ValueError('Missing required camera frame: '+row['scene_id'])
            images,_,camera_info = self._read_rgb(z,prefix,meta,row['context'])
        source_hw = tuple(images.shape[-2:])
        images,shift,scale,_ = self.transform.preprocess_images(images)
        cameras = torch.from_numpy(DL3DVArchive._cameras(meta)[:required]).float()
        cameras = torch.linalg.inv(cameras[row['context'][0]])[None] @ cameras
        cameras[:,:3,3] *= self.scene_scale
        k = torch.from_numpy(np.stack([rectified_intrinsics(meta,source_hw,i) for i in range(required)]))
        k = torch.stack((k[:,0]*scale[0],k[:,1]*scale[1],
            (k[:,2]+shift[0])*scale[0],(k[:,3]+shift[1])*scale[1]),-1)
        validate_cameras(cameras,k)
        if images.shape != (len(row['context']),3,*self.size) or not torch.isfinite(images).all():
            raise ValueError('Invalid rectified selected context RGB')
        return {'context_images':images[None],'c2w':cameras,'intrinsics':k,'source_hw':source_hw,
                'camera_preprocessing':camera_info,'crop_shift_xy':list(shift),'crop_scale_xy':list(scale)}

    def load_gt(self,row,expected_source_hw=None):
        path = self.root/(row['scene_id']+'.zip')
        with zipfile.ZipFile(path) as z:
            prefix,meta = zip_layout(z,path)
            rgb,_,_ = self._read_rgb(z,prefix,meta,row['target'])
        if expected_source_hw is not None and tuple(rgb.shape[-2:]) != tuple(expected_source_hw):
            raise ValueError('Context/target source raster changed: '+row['scene_id'])
        images,_,_,_ = self.transform.preprocess_images(rgb)
        if images.shape != (len(row['target']),3,*self.size) or not torch.isfinite(images).all():
            raise ValueError('Invalid rectified benchmark GT RGB')
        return images

def save_contact_sheet(path,context,pred,gt,row,frames):
    from PIL import Image,ImageDraw
    h,w = pred.shape[-2:];label = 24
    target_position = {f:i for i,f in enumerate(row['target'])}
    cells = [(context[i],f'INPUT frame {f}') for i,f in enumerate(row['context'])]
    for f in frames:
        cells.extend(((gt[target_position[f]],f'GT frame {f}'),(pred[target_position[f]],f'PRED frame {f}')))
    sheet = Image.new('RGB',(4*w,math.ceil(len(cells)/4)*(h+label)),'white');draw = ImageDraw.Draw(sheet)
    for n,(tensor,title) in enumerate(cells):
        x,y = (n%4)*w,(n//4)*(h+label)
        draw.text((x+4,y+4),title,fill='black')
        rgb = (tensor.float().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
        sheet.paste(Image.fromarray(rgb),(x,y+label))
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)
