# DiTGS

Residual Cross-Attention feature readout for 3D Gaussian reconstruction.

A camera-conditioned FlashWorld/Wan transformer predicts clean image latents.
The frozen Wan VAE decoder supplies features to an adapter, and twelve TokenGS
blocks read two independent memories:

```text
real RGB -> frozen TokenGS encoder -> GS CA -> residual DiT CA -> SA / MLP
predicted latents -> frozen VAE decoder -> adapter -----------^
```

The main model uses two real-view grids and all eight generated feature grids,
independent source KV projections, and learned per-channel LayerScale.
Only this feature-readout model and three attention controls are included.
The Wan VAE is a pretrained component; there is no separate learned scene-VAE route.

## Installation

Assume Linux, Python 3.11/3.12, NVIDIA GPUs supporting BF16, and a compatible
CUDA toolkit/compiler. The reference training configuration uses 16 GPU processes,
one scene per process. The CUDA packaging below is separate from any vendor runtime.

```bash
git clone https://github.com/Qingzhongyun/DiTGS.git
cd DiTGS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel ninja
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e '.[renderer]' --no-build-isolation
```

Core versions:PyTorch 2.8.0,torchvision 0.23.0,Diffusers 0.35.1,
Transformers 4.56.2,gsplat 1.5.3 and LPIPS 0.1.4.
`pyproject.toml` pins the fused-ssim source revision. Build its CUDA extension
against the same PyTorch/CUDA environment. No FlashAttention installation is needed;
attention uses PyTorch SDPA.

## Pretrained dependencies

Place the following locally before training; large weights are not in Git.

| Component | Source | Default local path |
| --- | --- | --- |
| Wan2.2 TI2V-5B Diffusers snapshot | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers),revision `b8fff7315c768468a5333511427288870b2e9635` | `weights/wan/` |
| FlashWorld NVS pretraining checkpoint | [nightkiller/FlashWorldPretrain](https://huggingface.co/nightkiller/FlashWorldPretrain),revision `896f3964301821d0bdd92ccb9560fbf51b855599` | `weights/flashworld/pretrain.ckpt` |
| TokenGS official base1024 | [TokenGS weights](https://huggingface.co/jiaweir/tokengs),`checkpoints/dl3dv_base.safetensors` | `weights/tokengs/dl3dv_base.safetensors` |

Wan requires its`transformer/`, `vae/`, `text_encoder/`and`tokenizer/`directories.
Download the complete pinned snapshot using the Hugging Face CLI, for example:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --revision b8fff7315c768468a5333511427288870b2e9635 --local-dir weights/wan
hf download nightkiller/FlashWorldPretrain pretrain.ckpt --revision 896f3964301821d0bdd92ccb9560fbf51b855599 --local-dir weights/flashworld
```

Use the original base1024 TokenGS file, not the newer 4096-token or bottleneck model.
`ditgs/flashworld.py` extracts the 48-channel latent-velocity output and the
54 pretrained condition channels directly from the original FlashWorld checkpoint.
Its 1024-channel GS-feature output and reconstruction decoder are discarded.
An already converted matching 826-key safetensors transformer is also accepted.
The masked 512-token empty UMT5 prompt is cached on first training startup.
LPIPS-VGG weights are downloaded/cached by LPIPS/torchvision at first evaluation;
prepare the cache ahead of time on offline machines.

**Trained DiTGS weights are not included,and no public trained-weight download
is configured in this source release.** Training produces the required raw
`model.pt`; supplying that full-model package is sufficient for inference.
It includes the Wan/TokenGS parameters,empty prompt and all normalization buffers.

## DL3DV data

Obtain DL3DV-10K and its released 140-scene benchmark under the dataset terms.
The loader expects ZIP archives containing`transforms.json`and the referenced
RGB images,typically `images_4/`. Flat and scene-prefixed archives are supported.

```text
data/DL3DV/
  train/1K/<scene>.zip ... train/10K/<scene>.zip
  benchmark/<scene>.zip
```

`assets/splits/dl3dv_train.txt`contains the 10341 training archive identities in the
reference order,relative to`--data-root`. `dl3dv_2view.json`contains the released
TokenGS 2-context/50-target index,all 140 scenes. Its SHA256 is
`434c3e1bd9395cf7f0f2f0dc787d15a9bddd87865891fd5817fdfa6e711ad46d`.
No RGB,depth or checkpoint payload is redistributed.

Training and evaluation both scale intrinsics to the actual decoded raster,
rectify declared lens distortion,and use exact integer center-crop offsets.
They retain the same output K,first-context GS coordinate frame and 0.15 scene scale.
Train at 256x256;evaluate natively at 256x448. Do not compare these rectified-image
scores directly with historical scores computed on distorted images.

## Training

```bash
bash scripts/train.sh --output outputs/residual
```

This runs 10000 accepted warmup updates followed by 20000 NEW joint updates.
Warmup trains the adapter,source CA/KV/LayerScale and native GS decoder/head;
Wan,image/camera conditioning,VAE and real-image encoder stay frozen.
Joint restores its own warmup model/buffers,starts fresh Adam/EMA/LR,and unfreezes
Wan including conditioning. VAE and real-image encoder remain frozen.

New branches use LR 1e-4;native GS and Wan use 1e-5. Both stages render all 8 training
views. Joint optimizes FM+IMG+GS with unit weights;GS includes RGB,SSIM and visibility.
The JSON configs contain the remaining optimizer/loss settings.

Paths are configurable through `--data-root`, `--training-list`, `--wan-model-dir`,
`--flashworld-checkpoint`and`--tokengs-checkpoint`. `--stage warmup|joint`selects one
stage;`--nproc-per-node`changes the process count(default 16). Changing the global
batch size changes the reference recipe. Resubmitting the same command resumes
its own safe checkpoint. Recoverable bad training scenes are resampled;invalid
updates are rejected collectively without changing Adam/EMA. Successful completion
keeps warmup/joint finals and exports`outputs/residual/model.pt`for inference.
Use`--no-export-full`to skip that extra export.

## Standard1/2-view inference

```bash
bash scripts/infer.sh 1 --checkpoint outputs/residual/model.pt --output outputs/residual/dl3dv_1view
bash scripts/infer.sh 2 --checkpoint outputs/residual/model.pt --output outputs/residual/dl3dv_2view
```

The default uses one GPU with sufficient memory for the full model.
Each run scores all 140 scenes/7000targets and saves`summary.json`,per-rank target
metrics,and 140 qualitative sheets. Novel-only and input-overlap metrics are separate.
Defaults:raw weights,EulerNFE 1,seed 42,native 256x448. Target RGB is read only after
reconstruction and rendering. Failed/missing scenes cannot produce a complete result.

The 1-view protocol uses the **first context of the released 2-view index**,not an
independently released 1-view split;it has 6860 novel and 140 input-overlap targets.
The 2-view protocol has 6720 novel and 280 input-overlap targets. Training uses 2 inputs;
1-view inference is a view-count generalization setting. Both retain 8 DiT camera slots.

Pass`--data-root`for the benchmark ZIP directory,`--nfe`or`--seed`when needed.
A sharded final directory is also supported;set`NPROC_PER_NODE`to its original
process count. Full exports can be created separately without evaluation:

```bash
NPROC_PER_NODE=16 bash scripts/infer.sh 2 --checkpoint outputs/residual/joint --export-full outputs/residual/model.pt
```

## Attention controls

| Script | Definition |
| --- | --- |
| `scripts/train_single.sh` | One softmax over2 real + 6 missing-view grids,shared Q/K/V/output |
| `scripts/train_gated.sh` | Two source softmaxes,shared projections,per-layer`tanh(g)`initialized at 0 |
| `scripts/train_gate1.sh` | Same dual-softmax calculation,coefficient exactly 1 throughout training and evaluation |

Each script uses an independent default`outputs/<variant>/`directory and runs
warmup 10k -> joint 20k -> full 2-view evaluation. It uses the same corrected-camera
FlashWorld training recipe as the main model. These are retrained controls,
not test-time overrides of a learned-gate checkpoint. Their source policy replaces
observed adapter grids with true encoder features;the main residual model instead
keeps independent 2-view real and 8-view generated memories.

## License and citation

Code is released under **Apache-2.0**; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
TokenGS copyright and attribution are retained. External weights, datasets and
installed dependencies keep their own licenses. See [CITATION.cff](CITATION.cff)
for software citation;paper bibliographic details will be added when available.

This curated release has received static source, configuration, dependency-closure
and syntax checks. CUDA training/inference were not rerun as part of publication.
