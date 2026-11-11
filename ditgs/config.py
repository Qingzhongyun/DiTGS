"""Portable configuration for Residual CA and three shared-attention controls."""
from dataclasses import asdict,dataclass,fields
import hashlib,json,math
from pathlib import Path
from tokengs.camera_policy import CAMERA_PREPROCESSING
MODEL_REPO="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MODEL_REVISION="b8fff7315c768468a5333511427288870b2e9635"
PROTOCOL="ditgs-flashworld-seqca-layerscale-camera-fixed-full8-v7"
SOURCE_ATTENTION="sequential-residual-ca-independent-source-kv"
QUALITY_SHA256=""
VARIANTS={
    "single":("single-softmax-shared-native-projections","none"),
    "gated":("shared-projections-dual-softmax-tanh-zero-gate","learned-per-layer-tanh-zero-initialized"),
    "gate1":("shared-projections-dual-softmax-fixed-one","exact-one-throughout-training-and-evaluation"),
}

def pfs_path(value,name):
    """Resolve a user-supplied path (no cluster-specific root restriction)."""
    if not value:raise ValueError(name+" must be specified")
    return Path(value).expanduser().resolve()

@dataclass
class Config:
    protocol: str = PROTOCOL
    model_repo: str = MODEL_REPO
    model_revision: str = MODEL_REVISION
    vae_temporal_policy: str = 'crop-causal-history-t1'
    feature_source: str = 'vae-decoder-final-norm-activation'
    adapter_patch_size: int = 4
    ray_encoding: str = 'tokengs-moment-direction'
    recipe: str = 'gs_warmup'
    camera_preprocessing: str = CAMERA_PREPROCESSING
    conditioning_projection: str = 'split-pretrained-flashworld-native-condition'
    backbone_source: str = 'nightkiller/FlashWorldPretrain'
    flashworld_revision: str = '896f3964301821d0bdd92ccb9560fbf51b855599'
    flashworld_source_sha256: str = '5da5eb36fb5193cd057b04239e3576680512e998888eb64bfc195617df4e243c'
    flashworld_transformer: str = 'weights/flashworld/pretrain.ckpt'
    latent_normalization: str = 'flashworld-pretrain-native-bf16-rounded'
    dit_camera_policy: str = 'flashworld-opengl-firstcam-maxradius-v1'
    dit_camera_normalization_eps: float = 0.01
    dit_output_initialization: str = 'copy-native-ca-unfolded'
    residual_post_norm: str = 'none'
    dit_residual_scaling: str = 'native-trainable-per-channel-layerscale'
    dit_layerscale_initialization: str = 'copy-native-gs-ca-gamma-independent'
    dit_kv_initialization: str = 'copy-native-kv-and-keynorm-independent'
    adapter_norm_initialization: str = 'copy-native-encoder-final-ln'
    visibility_policy: str = 'unculled-frustum-log-distance-v1'
    training_policy: str = 'camera-fixed-adapter-gs-warmup10k-then-joint20k'
    max_without_renderer_update: int = 128
    max_without_gs_update: int = 128
    gaussian_policy: str = 'local-repair-v1'
    view_fusion: str = 'separate-2view-gs-8view-dit-memories'
    sparse_encoder_policy: str = 'frozen-official-base1024-joint-two-view'
    source_attention: str = SOURCE_ATTENTION
    kv_sharing_policy: str = 'independent-sources-shared-across-decoder-layers'
    gs_memory_views: int = 2
    dit_memory_views: int = 8
    warmup_trainability: str = 'adapter-ca-kv-layerscale-and-native-gs-decoder-head-wan-frozen'
    gs_render_policy: str = 'all-eight-views-both-observed-and-six-missing'
    data_recovery_policy: str = 'unbounded-real-scene-resampling'
    init_checkpoint: str = ''
    data_replacement_attempts: int = 8
    focal_ratio_min: float = 0.01
    focal_ratio_max: float = 100.0
    camera_translation_max: float = 100.0
    camera_baseline_min: float = 1e-08
    latent_abs_max: float = 1000.0
    feature_abs_max: float = 10000.0
    gaussian_abs_max: float = 1000.0
    loss_abs_max: float = 10000.0
    gradient_norm_min: float = 1e-12
    gradient_norm_max: float = 1000000.0
    recovery_backoff_seconds: float = 10.0
    img_weight: float = 1.0
    asset_root: str = 'weights'
    model_dir: str = 'weights/wan'
    workspace: str = 'outputs/residual/warmup'
    official_checkpoint: str = 'weights/tokengs/dl3dv_base.safetensors'
    data_root: str = 'data/DL3DV/train'
    benchmark_root: str = 'data/DL3DV/benchmark'
    quality_manifest: str = ''
    quality_manifest_sha256: str = ''
    image_size: int = 256
    image_width: int = 256
    dense_views: int = 8
    gs_render_views: int = 8
    scene_scale: float = 0.15
    min_gap: int = 7
    max_gap: int = 40
    feature_dim: int = 1024
    seed: int = 42
    steps: int = 10000
    batch_size: int = 1
    world_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    save_every: int = 5000
    log_every: int = 10
    mixed_precision: str = 'bf16'
    distributed: str = 'fsdp'
    activation_checkpointing: bool = True
    vae_view_batch: int = 1
    lr_dit: float = 1e-05
    lr_gs: float = 0.0001
    lr_original_gs: float = 1e-05
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    optimizer_betas: tuple = (0.9, 0.95)
    ema_decay: float = 0.9999
    fm_weight: float = 0.0
    gs_weight: float = 1.0
    ssim_weight: float = 0.2
    visibility_weight: float = 0.01
    visibility_distance_scale: float = 0.15
    diagnostic_spike_ratio: float = 100.0
    diagnostic_capture_every: int = 100
    resume: str = ''
    variant: str="residual"
    training_list: str="assets/splits/dl3dv_train.txt"
    attention_variant: str=""
    completion_gate_policy: str="none"

    def __post_init__(self):
        self.optimizer_betas=tuple(self.optimizer_betas)
        if self.attention_variant and self.variant=="residual":
            self.variant=self.attention_variant
        if self.variant in VARIANTS:
            self.attention_variant=self.variant
            self.source_attention,self.completion_gate_policy=VARIANTS[self.variant]
            self.protocol=f"ditgs-flashworld-{self.variant}-ca-camera-fixed-2real6generated-v1"
            self.training_policy=f"camera-fixed-{self.variant}-ca-warmup10k-then-joint20k"
            self.view_fusion="hard-observed-slot-replacement-2real6generated"
            self.kv_sharing_policy="shared-between-sources-and-across-decoder-layers"
            self.dit_memory_views=6
            self.dit_output_initialization="official-native-shared-ca"
            self.dit_residual_scaling="native-shared-ca-trainable-per-channel-layerscale"
            self.dit_layerscale_initialization="official-native-shared-ca-gamma"
            self.dit_kv_initialization="official-native-shared-kv-and-keynorm"
            self.warmup_trainability="adapter-native-shared-ca-kv-decoder-head-and-gate-if-present-wan-frozen"

    @property
    def freeze_dit(self):return self.recipe=="gs_warmup"
    @property
    def freeze_original_gs(self):return False
    @property
    def train_dit_condition(self):return not self.freeze_dit
    @property
    def lineage_offset(self):return 0 if self.recipe=="gs_warmup" else 10000
    @property
    def use_gs(self):return True
    @property
    def use_img(self):return self.recipe=="joint"
    @property
    def warmup_workspace(self):return str(Path(self.workspace).resolve().parent/"warmup")

    def validate(self,check_files=False):
        if self.variant not in ("residual",*VARIANTS):raise ValueError("Unknown model variant")
        if self.recipe not in ("gs_warmup","joint") or self.steps!=(10000 if self.recipe=="gs_warmup" else 20000):
            raise ValueError("Use10000warmup and20000new joint accepted updates")
        if self.camera_preprocessing!=CAMERA_PREPROCESSING:raise ValueError("Camera policy mismatch")
        if (self.dense_views,self.gs_render_views,self.gs_memory_views)!=(8,8,2):raise ValueError("Training requires two real/eight total views and full8renders")
        if self.feature_dim!=1024 or self.scene_scale!=.15:raise ValueError("Native TokenGS dimensions/scale required")
        if self.dit_memory_views!=(8 if self.variant=="residual" else 6):raise ValueError("Source-memory policy mismatch")
        if (self.fm_weight,self.gs_weight,self.img_weight)!=((0. if self.freeze_dit else 1.),1.,1.):raise ValueError("Objective schedule mismatch")
        if min(self.world_size,self.batch_size,self.num_workers+1,self.save_every,self.log_every,self.warmup_steps)<=0:raise ValueError("Invalid training counts")
        if self.batch_size!=1:raise ValueError("The released per-rank recipe uses batch_size1")
        if self.image_size%16 or self.image_width%16:raise ValueError("Raster dimensions must be multiples of16")
        if self.min_gap<7 or self.max_gap<self.min_gap:raise ValueError("Invalid frame intervals")
        if not 0<self.lr_original_gs<self.lr_gs or self.lr_dit<=0:raise ValueError("Invalid learning rates")
        numeric=(self.lr_gs,self.lr_original_gs,self.lr_dit,self.gradient_clip,self.weight_decay,self.ema_decay,self.visibility_weight,self.visibility_distance_scale)
        if not all(math.isfinite(x) for x in numeric):raise ValueError("Nonfinite coefficient")
        if self.recipe=="gs_warmup" and self.init_checkpoint:raise ValueError("Warmup starts from upstream pretrained assets")
        if self.resume and Path(self.resume).resolve().parent!=Path(self.workspace).resolve()/"checkpoints":raise ValueError("Resume must belong to its own stage")
        if check_files:
            for name in ("training_list","official_checkpoint","flashworld_transformer"):
                if not Path(getattr(self,name)).is_file():raise FileNotFoundError(getattr(self,name))
        return self

    def to_dict(self):return asdict(self)
    def recipe_hash(self):
        data=self.to_dict();data.pop("resume",None)
        return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

    def model_contract(self):
        record = {key: getattr(self, key) for key in ('protocol', 'model_repo', 'model_revision', 'dense_views', 'feature_dim', 'scene_scale', 'image_size', 'image_width', 'vae_temporal_policy', 'feature_source', 'adapter_patch_size', 'ray_encoding', 'conditioning_projection', 'gaussian_policy', 'view_fusion', 'sparse_encoder_policy', 'source_attention', 'data_recovery_policy', 'kv_sharing_policy', 'gs_memory_views', 'dit_memory_views', 'dit_residual_scaling', 'dit_layerscale_initialization', 'warmup_trainability', 'gs_render_views', 'gs_render_policy', 'backbone_source', 'flashworld_revision', 'flashworld_source_sha256', 'latent_normalization', 'dit_camera_policy', 'dit_camera_normalization_eps', 'dit_output_initialization', 'visibility_policy', 'visibility_distance_scale', 'training_policy', 'residual_post_norm', 'dit_kv_initialization', 'adapter_norm_initialization', 'camera_preprocessing')}
        if self.variant!="residual":
            record.update(attention_variant=self.attention_variant,completion_gate_policy=self.completion_gate_policy)
        return record

def config_from_dict(record):
    record=dict(record)
    known={f.name for f in fields(Config)}
    unknown=set(record)-known
    if unknown:raise ValueError(f"Unsupported configuration fields:{sorted(unknown)}")
    if record.get("attention_variant") and not record.get("variant"):
        record["variant"]=record["attention_variant"]
    return Config(**record).validate()

def config_from_json(path):return config_from_dict(json.loads(Path(path).read_text()))

def parse_config(argv=None):
    import argparse
    p=argparse.ArgumentParser(description="Distributed DiTGS training")
    p.add_argument("--config",required=True);p.add_argument("--resume");p.add_argument("--manual-resume",default="")
    a=p.parse_args(argv);cfg=config_from_json(a.config)
    if a.resume is not None:cfg.resume=a.resume
    cfg._manual_resume=bool(a.manual_resume) or bool(a.resume)
    return cfg.validate()
