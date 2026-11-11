#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: bash scripts/infer.sh 1|2 --checkpoint PATH [options]" >&2; exit 64; }
VIEWS="$1";shift
case "$VIEWS" in 1|2) ;; *) echo "Input views must be1or2" >&2;exit 64;; esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nnodes=1   --nproc_per_node="${NPROC_PER_NODE:-1}" --master_port="${MASTER_PORT:-29500}"   -m ditgs.inference --input-views "$VIEWS" "$@"
