"""Same RGB metric definitions as the fixed DL3DV evaluator."""
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from skimage.metrics import structural_similarity

# FP32 antialiased resize can overshoot an endpoint by a few ULPs.
GT_ROUNDOFF_ATOL = 8 * torch.finfo(torch.float32).eps

@torch.no_grad()
def score_images(pred,gt,perceptual,device,chunk=4):
    if pred.shape!=gt.shape or pred.ndim!=4 or pred.shape[1]!=3:
        raise ValueError("Prediction/GT raster mismatch")
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError("Nonfinite predicted/target RGB")
    low,high=float(gt.min()),float(gt.max())
    if low < -GT_ROUNDOFF_ATOL or high > 1+GT_ROUNDOFF_ATOL:
        raise ValueError(f"GT outside FP32 resize tolerance: min={low}, max={high}, atol={GT_ROUNDOFF_ATOL}")
    pred=pred.float().clamp(0,1);gt=gt.float()
    scores=[]
    for offset in range(0,len(pred),chunk):
        p=pred[offset:offset+chunk];g=gt[offset:offset+chunk]
        # Match the existing official evaluator's PSNR endpoint handling.
        # Keep preprocessed GT unmodified for SSIM/LPIPS.
        mse=(p-g.clamp(0,1)).square().mean((1,2,3))
        psnr=-10*mse.log10()
        ssim=[float(structural_similarity(x.numpy(),y.numpy(),win_size=11,
             gaussian_weights=True,channel_axis=0,data_range=1.)) for x,y in zip(g,p)]
        with torch.autocast(device_type=torch.device(device).type,enabled=False):
            lp=perceptual(g.to(device),p.to(device),normalize=True).flatten().float().cpu()
        for i in range(len(p)):
            result={"psnr":float(psnr[i]),"ssim":ssim[i],"lpips":float(lp[i])}
            if not all(math.isfinite(v) for v in result.values()):
                raise FloatingPointError("Nonfinite benchmark metric")
            scores.append(result)
    return scores

def save_contact_sheet(path,context,pred,gt,row,frames):
    h,w=pred.shape[-2:];label=24
    sheet=Image.new("RGB",(4*w,4*(h+label)),"white")
    draw=ImageDraw.Draw(sheet)
    cells=[(context[i],f"INPUT frame {f}") for i,f in enumerate(row["context"])]
    for f in frames:
        if f not in row["context"]:
            cells.extend(((gt[f],f"GT frame {f}"),(pred[f],f"PRED frame {f}")))
    def as_image(t):
        return Image.fromarray((t.float().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8))
    for n,(tensor,title) in enumerate(cells):
        x=(n%4)*w;y=(n//4)*(h+label)
        draw.text((x+4,y+4),title,fill="black")
        sheet.paste(as_image(tensor),(x,y+label))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)
