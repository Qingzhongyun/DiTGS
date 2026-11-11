"""Two-stage training and the three optional train/evaluate controls."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .config import config_from_json
from .launch_setup import resolve_launch_config,write_launch_config,validate_completed_phase

def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("train","ablation"))
    p.add_argument("--variant",choices=("residual","single","gated","gate1"),default="residual")
    p.add_argument("--stage",choices=("both","warmup","joint"),default="both")
    p.add_argument("--config-dir",default="configs")
    p.add_argument("--output")
    p.add_argument("--data-root",default="data/DL3DV/train")
    p.add_argument("--benchmark-root",default="data/DL3DV/benchmark")
    p.add_argument("--training-list",default="assets/splits/dl3dv_train.txt")
    p.add_argument("--wan-model-dir",default="weights/wan")
    p.add_argument("--flashworld-checkpoint",default="weights/flashworld/pretrain.ckpt")
    p.add_argument("--tokengs-checkpoint",default="weights/tokengs/dl3dv_base.safetensors")
    p.add_argument("--nproc-per-node",type=int,default=int(os.environ.get("NPROC_PER_NODE",16)))
    p.add_argument("--master-port",type=int,default=int(os.environ.get("MASTER_PORT",29500)))
    p.add_argument("--max-process-restarts",type=int,default=12)
    p.add_argument("--export-full",action=argparse.BooleanOptionalAction,default=True)
    return p.parse_args(argv)

def torchrun(args,module,options):
    return [sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",f"--master_port={args.master_port}",
        "--max_restarts=0","--monitor_interval=5","-m",module,*options]

def main(argv=None):
    args=arguments(argv)
    if args.nproc_per_node<1 or not 1024<=args.master_port<=65535:raise ValueError("Invalid process count/port")
    output=Path(args.output or "outputs/"+args.variant).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/".train.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=("warmup","joint") if args.stage=="both" else (args.stage,)
        for stage in stages:
            cfg=config_from_json(Path(args.config_dir)/(stage+".json"))
            cfg.variant=args.variant;cfg.attention_variant="";cfg.__post_init__()
            cfg.workspace=str(output/stage);cfg.world_size=args.nproc_per_node
            for key,value in (("data_root",args.data_root),("benchmark_root",args.benchmark_root),("training_list",args.training_list),
                              ("model_dir",args.wan_model_dir),("flashworld_transformer",args.flashworld_checkpoint),("official_checkpoint",args.tokengs_checkpoint)):
                setattr(cfg,key,str(Path(value).expanduser().resolve()))
            cfg=resolve_launch_config(cfg)
            config=write_launch_config(cfg)
            if (Path(cfg.workspace)/"completion.json").is_file():
                validate_completed_phase(cfg);print(f"{stage} already complete",flush=True);continue
            previous_safe=None;unchanged=0
            for attempt in range(args.max_process_restarts+1):
                env=dict(os.environ,DITGS_RECOVERY_RESTART=str(attempt),PYTHONDONTWRITEBYTECODE="1")
                result=subprocess.run(torchrun(args,"ditgs.train",["--config",str(config),"--manual-resume",""]),env=env)
                if result.returncode==0:break
                workspace=Path(cfg.workspace);pointer=workspace/"latest_safe.json"
                if result.returncode in (130,143,-2,-15) or (workspace/"training_failed.json").exists() or not pointer.exists():raise SystemExit(result.returncode)
                safe=pointer.read_text();unchanged=unchanged+1 if safe==previous_safe else 0;previous_safe=safe
                if unchanged>=2 or attempt>=args.max_process_restarts:raise SystemExit(result.returncode)
                time.sleep(min(10*(attempt+1),60))
            cfg.resume=""
            cfg=resolve_launch_config(cfg);validate_completed_phase(cfg)
        joint=output/"joint"
        if args.export_full and (joint/"completion.json").is_file() and not (output/"model.pt").exists():
            subprocess.run(torchrun(args,"ditgs.inference",["--checkpoint",str(joint),"--export-full",str(output/"model.pt")]),check=True)
        if args.command=="ablation":
            if not (joint/"completion.json").is_file():raise ValueError("Ablation evaluation requires joint completion")
            run=output/("dl3dv_2view_"+time.strftime("%Y%m%dT%H%M%S"))
            subprocess.run(torchrun(args,"ditgs.inference",["--checkpoint",str(joint),"--input-views","2","--data-root",str(Path(args.benchmark_root).resolve()),
                "--output",str(run),"--nfe","1","--seed","42","--save-scenes","140"]),check=True)

if __name__=="__main__":main()
