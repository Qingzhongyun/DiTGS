"""Standard fixed-DL3DV1/2-view reconstruction and full raw-weight export."""
import argparse,datetime,json,os
from pathlib import Path

def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--input-views",type=int,choices=(1,2),default=2)
    p.add_argument("--data-root",default="data/DL3DV/benchmark")
    p.add_argument("--index",default=str(Path(__file__).resolve().parents[1]/"assets/splits/dl3dv_2view.json"))
    p.add_argument("--output",default="outputs/inference")
    p.add_argument("--model-dir")
    p.add_argument("--nfe",type=int,default=1);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--save-scenes",type=int,default=140)
    p.add_argument("--export-full",help="Export a full raw model and exit without model forward or evaluation")
    return p.parse_args(argv)

def main(argv=None):
    args=arguments(argv)
    import copy,time
    import torch,torch.distributed as dist
    from .weights import load_model,export_full
    from .inference_runtime import MultiviewBenchmarkData,sample_latents,save_contact_sheet
    from .sampling_gs import render_scene,finite_or_raise
    from .eval_protocol import index_rows,round_plan,seed_for,aggregate_records
    from .eval_metrics import score_images
    from tokengs.rendering.gs import GaussianRenderer
    rank,world,local=int(os.environ.get("RANK",0)),int(os.environ.get("WORLD_SIZE",1)),int(os.environ.get("LOCAL_RANK",0))
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    if world>1:dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=40))
    def agree(error,label):
        errors=[error]
        if world>1:errors=[None]*world;dist.all_gather_object(errors,error)
        if any(errors):raise RuntimeError(label+":"+";".join(f"rank{i}:{e}" for i,e in enumerate(errors) if e))
    def coordinated(fn,label):
        out=None;err=None
        try:out=fn()
        except Exception as e:err=f"{type(e).__name__}:{e}"
        agree(err,label);return out
    def finite(name,value):
        flag=torch.isfinite(value).all().int()
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        if not bool(flag):raise FloatingPointError(name)
    try:
        raw,model,cfg,sharded=load_model(args.checkpoint,device,rank,world,args.model_dir)
        if args.export_full:
            coordinated(lambda:export_full(raw,model,cfg,args.export_full,rank,sharded),"export raw model")
            if world>1:dist.barrier()
            return
        if not 1<=args.nfe<=256 or not 0<=args.save_scenes<=140:raise ValueError("Invalid inference options")
        raw.configure_views(args.input_views)
        baseline=index_rows(args.index)
        rows=[{**row,"baseline_context":row["context"],"context":row["context"][:args.input_views]} for row in baseline]
        output=Path(args.output).expanduser().resolve()
        coordinated(lambda:output.mkdir(parents=True,exist_ok=False) if rank==0 else None,"create output")
        data=MultiviewBenchmarkData(args.data_root,scene_scale=cfg.scene_scale)
        options=copy.copy(raw.gs.opt);options.img_size=(256,448)
        renderer=GaussianRenderer(options);background=raw.gs._background_color(torch.float32,device)
        import lpips
        perceptual=coordinated(lambda:lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False),"LPIPS-VGG")
        records=[]
        sink=(output/f"samples_rank{rank:03d}.jsonl").open("x")
        try:
            with torch.no_grad():
                for round_id,slots in enumerate(round_plan(len(rows),world)):
                    idx,active=slots[rank];row=rows[idx];sid=row["scene_id"]
                    case=coordinated(lambda:data.load_condition(row),"conditions:"+sid)
                    first,last=row["baseline_context"];frames=[round(first+(last-first)*i/7) for i in range(8)]
                    context_slots=[frames.index(i) for i in row["context"]]
                    c2w=case["c2w"].to(device);intrinsics=case["intrinsics"].to(device)
                    torch.cuda.synchronize();started=time.perf_counter()
                    with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
                        context=model("encode_context",images=case["context_images"].to(device),generator=torch.Generator(device=device).manual_seed(seed_for(args.seed,sid,"context","vae")))
                        finite("context",context)
                        state=sample_latents(model,context,c2w[frames][None],intrinsics[frames][None],frames,context_slots,
                            height=256,width=448,nfe=args.nfe,seed=args.seed,scene_id=sid,scene_scale=cfg.scene_scale,
                            normalization_eps=cfg.dit_camera_normalization_eps,check=finite)
                        points,valid,stats=model("reconstruct_gs",state=state,c2w=c2w[frames][None],intrinsics=intrinsics[frames][None],
                            observed_images=case["context_images"].to(device),observed_mask=torch.tensor([[i in context_slots for i in range(8)]],device=device,dtype=torch.bool),check=finite)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-started
                    def finish():
                        if not active:return None
                        pred=render_scene(renderer,points,valid,c2w,intrinsics,row["target"],background)
                        gt=data.load_gt(row,expected_source_hw=case["source_hw"])
                        scores=score_images(pred,gt,perceptual,device)
                        if idx<args.save_scenes:
                            novel=[f for f in row["target"] if f not in row["context"]]
                            chosen=[novel[round(i*(len(novel)-1)/5)] for i in range(6)]
                            save_contact_sheet(output/"qualitative"/f"{idx:03d}_{sid}.png",case["context_images"][0],pred,gt,row,chosen)
                        return {"sample_index":idx,"scene_id":sid,"context":row["context"],"target":row["target"],
                            "per_target":[{"frame":f,**score} for f,score in zip(row["target"],scores)],"reconstruction_seconds":elapsed,"gaussian_statistics":stats}
                    record=coordinated(finish,"render/score:"+sid)
                    if record is not None:sink.write(json.dumps(record,allow_nan=False)+"\n");sink.flush()
        finally:sink.close()
        if world>1:dist.barrier()
        def publish():
            if rank!=0:return
            records=[json.loads(line) for p in output.glob("samples_rank*.jsonl") for line in p.read_text().splitlines()]
            scores=aggregate_records(records,rows)
            expected_novel=7000-140*args.input_views
            if (scores["scenes"],scores["target_images"],scores["novel_target_images"])!=(140,7000,expected_novel):raise ValueError("Incomplete benchmark")
            inputs=[x for row in records for x in row["per_target"] if x["frame"] in row["context"]]
            input_metrics={k:sum(x[k] for x in inputs)/len(inputs) for k in ("psnr","ssim","lpips")}
            summary={**scores,"input_only_metrics":input_metrics,"input_views":args.input_views,"variant":cfg.variant,"checkpoint":str(args.checkpoint),
                "nfe":args.nfe,"seed":args.seed,"camera_preprocessing":cfg.camera_preprocessing,"resolution":[256,448],"weights":"raw",
                "one_view_protocol":"first released two-view context;all50 targets retained" if args.input_views==1 else None}
            (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
            (output/"COMPLETE").write_text("140 scenes /7000 targets\n")
            print(json.dumps(summary,indent=2),flush=True)
        coordinated(publish,"publish")
    finally:
        if world>1 and dist.is_initialized():dist.destroy_process_group()

if __name__=="__main__":main()
