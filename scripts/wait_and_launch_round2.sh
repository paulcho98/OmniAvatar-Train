#!/bin/bash
# Wait for round 1 of run_v2v_eval_ablation.sh to finish,
# kill it, then launch the balanced round 2.
set -euo pipefail

cd /home/work/.local/OmniAvatar

echo "Waiting for round 1 to finish (checking 2500_noffo/hdtf_composited)..."
while [ ! -d "demo_out/v2v_eval_ablation/2500_noffo/hdtf_composited" ] || \
      [ "$(ls demo_out/v2v_eval_ablation/2500_noffo/hdtf_composited/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; do
    sleep 30
done

echo "Round 1 done. Killing main script..."
pkill -f "run_v2v_eval_ablation.sh" || true
sleep 5

echo "Starting balanced round 2..."
exec bash scripts/run_v2v_eval_ablation_round2.sh 2>&1 | tee -a demo_out/v2v_eval_ablation.log
