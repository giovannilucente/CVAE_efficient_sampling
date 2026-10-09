#!/bin/bash
# Cost-aware CVAE training as a Slurm job. If the 24 h limit ends the job, submit the same script
# again: train_cost_cvae.py resumes from <run dir>/ckpt_last.pt.
# Adjust the partition / account lines to the cluster.
#SBATCH --job-name=cost_cvae
#SBATCH --gres=gpu:1                  # one A40 or A100 is plenty (about 2 GB GPU memory at batch 64)
#SBATCH --cpus-per-task=9             # 8 data-loading workers + main process
#SBATCH --mem=48G                     # the cache is memory-mapped; most of it is reclaimable page cache
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@600             # 10 min before the limit: checkpoint and stop cleanly
#SBATCH --output=cost_cvae_%j.log

CACHE=${CACHE:-$HOME/data/cem_train_R8x250_cache}   # copy of the cache built by cem_cache.py
RUN=${RUN:-$HOME/runs/cost_cvae_tau0.5}

srun python train_cost_cvae.py --cache "$CACHE" --out "$RUN" --workers 8 --max_hours 23.7 "$@"
