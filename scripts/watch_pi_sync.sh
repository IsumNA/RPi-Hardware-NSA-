#!/usr/bin/env bash
# Poll AI Pi cache sync + project.json readiness; when HCG manifest coverage
# exceeds 50%, print (or optionally run) align_pi_cache + HCG train_raw_visual.
#
# Usage (laptop):
#   ./scripts/watch_pi_sync.sh
#   ./scripts/watch_pi_sync.sh --interval 60
#   ./scripts/watch_pi_sync.sh --execute   # run long jobs on AI when ready
#
# Environment:
#   NSA_AI_HOST       SSH host (default: ai)
#   PI_UNIQUE_CACHE   on AI (default: /opt/datasets/PI_RAW/Pi_Unique_Cache)
set -euo pipefail

REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
AI="${NSA_AI_HOST:-ai}"
REMOTE_REPO='~/RPi-Hardware-NSA-'
PI_CACHE="${PI_UNIQUE_CACHE:-/opt/datasets/PI_RAW/Pi_Unique_Cache}"
INTERVAL=30
THRESHOLD=50
EXECUTE=0
ONCE=0

usage() {
  cat <<'EOF'
watch_pi_sync.sh — poll AI HCG sync until ready for alignment + HCG training

Options:
  --interval SEC   poll period (default 30)
  --threshold PCT  coverage to trigger (default 50)
  --execute        run align_pi_cache --all + train_raw_visual on AI (default: print only)
  --once           single readiness check, then exit
  -h, --help       this message

When coverage > threshold and project.json is present, prints:
  ssh ai 'cd ~/RPi-Hardware-NSA- && python align_pi_cache.py --all'
  ssh ai 'cd ~/RPi-Hardware-NSA- && nohup .venv/bin/python -u train_raw_visual.py ...'
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --once) ONCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

readiness_cmd() {
  cat <<REMOTE
cd ${REMOTE_REPO}
.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path

cache = Path("${PI_CACHE}")
repo = Path.home() / "RPi-Hardware-NSA-"
sys.path.insert(0, str(repo))

from nsa.dataset_align import (
    build_hcg_sort_manifest,
    cache_readiness,
    project_json_in_cache,
)

pj = project_json_in_cache(cache)
out = {
    "project_json": str(pj),
    "project_json_present": pj.is_file(),
    "manifest_ready": False,
    "coverage_pct": 0.0,
    "present_files": 0,
    "wanted_files": 0,
    "status": "waiting",
}

if not pj.is_file():
    print(json.dumps(out))
    sys.exit(0)

manifest = build_hcg_sort_manifest(pj)
out["manifest_ready"] = True
report = cache_readiness(cache, manifest)
out.update({
    "coverage_pct": round(100.0 * report["fraction"], 2),
    "present_files": report["present_files"],
    "wanted_files": report["wanted_files"],
    "status": "ready" if report["fraction"] * 100.0 >= ${THRESHOLD} else "syncing",
})
print(json.dumps(out))
PY
REMOTE
}

print_ready_commands() {
  local cov="$1"
  echo ""
  echo "==> HCG sync ${cov}% (>= ${THRESHOLD}%) — project.json + manifest ready"
  echo ""
  echo "# 1) Align Pi cache → HCG bursts + PI_RAW pairs (on AI):"
  echo "ssh -o BatchMode=yes ${AI} 'cd ${REMOTE_REPO} && .venv/bin/python align_pi_cache.py --all'"
  echo ""
  echo "# 2) Train RawDenoiser on HCG bursts (on AI GPU):"
  echo "ssh -o BatchMode=yes ${AI} bash -s <<'REMOTE'"
  echo "set -euo pipefail"
  echo "cd ${REMOTE_REPO}"
  echo "mkdir -p outputs/train_logs"
  echo "STAMP=\$(date +%Y%m%d-%H%M%S)"
  echo "LOG=outputs/train_logs/train_raw_visual_hcg_\${STAMP}.log"
  echo "nohup .venv/bin/python -u train_raw_visual.py \\"
  echo "  --burst-scene cabinet_H_2 \\"
  echo "  --gains 128,256,512 \\"
  echo "  --panel-every 50 \\"
  echo "  --panel-dir outputs/raw_panels \\"
  echo "  --force \\"
  echo "  >\"\$LOG\" 2>&1 &"
  echo "echo TRAIN_PID=\$!"
  echo "echo TRAIN_LOG=\$LOG"
  echo "REMOTE"
  echo ""
}

run_on_ai() {
  echo "==> Executing align_pi_cache --all on ${AI}"
  ssh -o BatchMode=yes "${AI}" "cd ${REMOTE_REPO} && .venv/bin/python align_pi_cache.py --all"

  echo "==> Starting train_raw_visual (HCG) on ${AI}"
  ssh -o BatchMode=yes "${AI}" bash -s <<REMOTE
set -euo pipefail
cd ${REMOTE_REPO}
mkdir -p outputs/train_logs
STAMP=\$(date +%Y%m%d-%H%M%S)
LOG=outputs/train_logs/train_raw_visual_hcg_\${STAMP}.log
nohup .venv/bin/python -u train_raw_visual.py \
  --burst-scene cabinet_H_2 \
  --gains 128,256,512 \
  --panel-every 50 \
  --panel-dir outputs/raw_panels \
  --force \
  >"\$LOG" 2>&1 &
echo TRAIN_PID=\$!
echo TRAIN_LOG=\$LOG
REMOTE
}

poll_once() {
  local json
  if ! json="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${AI}" "$(readiness_cmd)" 2>/dev/null)"; then
    echo "$(date +%H:%M:%S) WARN: cannot reach ${AI}"
    return 1
  fi

  local pj_present manifest_ready cov present wanted status
  pj_present="$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('project_json_present', False))")"
  manifest_ready="$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('manifest_ready', False))")"
  cov="$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('coverage_pct', 0))")"
  present="$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('present_files', 0))")"
  wanted="$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('wanted_files', 0))")"
  status="$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status', '?'))")"

  echo "$(date +%H:%M:%S) AI sync: ${cov}% (${present}/${wanted} manifest files)  project.json=${pj_present}  status=${status}"

  if [[ "$manifest_ready" == "True" && "$pj_present" == "True" ]]; then
    local cov_int
    cov_int="$(python3 -c "print(int(float('${cov}')))")"
    if [[ "$cov_int" -ge "$THRESHOLD" ]]; then
      if [[ "$EXECUTE" -eq 1 ]]; then
        run_on_ai
      else
        print_ready_commands "$cov"
      fi
      return 0
    fi
  fi
  return 1
}

chmod +x "${BASH_SOURCE[0]}" 2>/dev/null || true

echo "Watching ${AI} Pi cache (${PI_CACHE}) — threshold ${THRESHOLD}%  interval ${INTERVAL}s"
echo "Repo: ${REPO_LOCAL}  (execute=${EXECUTE})"
echo "Ctrl+C to stop"
echo ""

if [[ "$ONCE" -eq 1 ]]; then
  poll_once || true
  exit 0
fi

while true; do
  if poll_once; then
    if [[ "$EXECUTE" -eq 0 ]]; then
      echo ""
      echo "Threshold met — commands printed above. Re-run with --execute to launch on AI."
    fi
    exit 0
  fi
  sleep "$INTERVAL"
done
