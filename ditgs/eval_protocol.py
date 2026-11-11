"""Fixed DL3DV protocol and deterministic 8-view query groups; no torch imports."""
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

INDEX_SHA256 = "434c3e1bd9395cf7f0f2f0dc787d15a9bddd87865891fd5817fdfa6e711ad46d"
SCENES, TARGETS = 140, 7000
HEIGHT, WIDTH = 256, 448
BENCHMARK = Path(__file__).resolve().parents[1]/"assets/splits"

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def index_rows(path, official_tree=None):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INDEX_SHA256:
        raise ValueError("Changed released TokenGS 2-view index")
    entries = json.loads(raw)
    official=set(entries)
    if len(entries) != SCENES or not set(entries).issubset(official):
        raise ValueError("Incomplete or nonofficial scene set")
    rows = []
    for scene_id, row in entries.items():
        if (len(row["context"]) != 2 or row["context"] != sorted(set(row["context"]))
                or not all(type(i) is int and 0 <= i < 50 for i in row["context"])
                or row["target"] != list(range(50))):
            raise ValueError("Changed context/target protocol: " + scene_id)
        rows.append({"scene_id": scene_id, **row})
    return rows

def zip_layout(z, path):
    for name, prefix in (("transforms.json", ""),
            (path.stem+"/transforms.json", path.stem+"/"),
            (path.stem+"/gaussian_splat/transforms.json", path.stem+"/gaussian_splat/")):
        try:
            return prefix, json.loads(z.read(name))
        except KeyError:
            pass
    raise FileNotFoundError("No supported transforms.json: " + str(path))



def round_plan(count, world, verify_only=False):
    if count <= 0 or world <= 0:
        raise ValueError("Positive scene/world count required")
    if verify_only:
        return [[(0, rank == 0) for rank in range(world)]]
    return [[(round_id*world+rank, True) if round_id*world+rank < count else (0, False)
             for rank in range(world)] for round_id in range(math.ceil(count/world))]

def seed_for(base, scene_id, item, stream):
    raw = json.dumps([int(base), scene_id, item, stream], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63-1)

def aggregate_records(records, rows):
    ordered = sorted(records, key=lambda r:r["sample_index"])
    if [r["sample_index"] for r in ordered] != list(range(len(rows))):
        raise ValueError("Missing/duplicate scenes; no summary")
    for record, row in zip(ordered, rows):
        if any(record[k] != row[k] for k in ("scene_id","context","target")):
            raise ValueError("Changed benchmark scene/frame coverage")
        if [r["frame"] for r in record["per_target"]] != row["target"]:
            raise ValueError("Missing/duplicate targets; no summary")
        for r in record["per_target"]:
            if not all(math.isfinite(r[k]) for k in ("psnr","ssim","lpips")):
                raise ValueError("Nonfinite metric; no summary")
    def mean_metric(key, novel):
        return sum(sum(r[key] for r in scene["per_target"]
                       if not novel or r["frame"] not in scene["context"])/
                   sum(not novel or r["frame"] not in scene["context"]
                       for r in scene["per_target"]) for scene in ordered)/len(ordered)
    return {"scenes":len(rows), "target_images":sum(len(r["target"]) for r in rows),
            "novel_target_images":sum(len(set(r["target"])-set(r["context"])) for r in rows),
            "metrics":{k:mean_metric(k,False) for k in ("psnr","ssim","lpips")},
            "novel_only_metrics":{k:mean_metric(k,True) for k in ("psnr","ssim","lpips")}}
