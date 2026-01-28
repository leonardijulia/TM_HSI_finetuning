#!/bin/bash
# Sweep script for all experiments
# Submits 4 experiments × 2 band selections × 10 seeds = 80 jobs total

set -e

EXPERIMENTS=("enmap_cdl" "enmap_bnetd" "enmap_bdforet" "hyperview_1")
BAND_SELECTIONS=("naive" "srf_grouping")
SEEDS=(42 43 44 45 46 47 48 49 50 51)

TOTAL_JOBS=$((${#EXPERIMENTS[@]} * ${#BAND_SELECTIONS[@]} * ${#SEEDS[@]}))
echo "Submitting $TOTAL_JOBS total jobs"
echo "Experiments: ${EXPERIMENTS[@]}"
echo "Band selections: ${BAND_SELECTIONS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""

JOB_COUNT=0

for exp in "${EXPERIMENTS[@]}"; do
  for band_sel in "${BAND_SELECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOB_COUNT=$((JOB_COUNT + 1))
      echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting: experiments=$exp, band_selection=$band_sel, seed=$seed"
      sbatch scripts/train.sh $exp data.init_args.band_selection=$band_sel seed_everything=$seed
      # small delay to avoid overwhelming scheduler
      sleep 0.1
    done
  done
done

echo ""
echo "All $TOTAL_JOBS jobs submitted!"
