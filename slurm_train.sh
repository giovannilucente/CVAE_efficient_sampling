#!/bin/bash -l
# Cost-aware CVAE training as a Slurm job. Submit from CVAE_efficient_sampling/:
#     sbatch slurm_train.sh                                   # full training
#     sbatch slurm_train.sh --tau 0.25 --beta_end 0.1         # extra arguments go to train_cost_cvae.py
#     RUN=$WORK/runs/smoke sbatch slurm_train.sh --limit_scenarios 300 --epochs 2 --draws_per_cycle 4
# Shortly before the time limit Slurm sends SIGUSR1: the training writes a checkpoint and exits with
# code 3, and this script submits itself again (at most MAX_RESUBMIT times); the new job resumes from
# $RUN/ckpt_last.pt at the same position. Exit code 0 = training complete.
#
# ---- adjust to the cluster ---------------------------------------------------------------------
#SBATCH --job-name=cost_cvae
#SBATCH --partition=a40               # NHR@FAU alex; for an A100: --partition=a100 --gres=gpu:a100:1
#SBATCH --gres=gpu:a40:1              # one GPU (about 2 GB GPU memory at batch 64)
# no --mem / --cpus-per-task on alex: CPUs and RAM are allocated with the GPU
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@600             # 10 min before the limit: checkpoint and stop cleanly
#SBATCH --output=cost_cvae_%j.log
CACHE=${CACHE:-$WORK/cem_train_cache}              # cache built by cem_cache.py
RUN=${RUN:-$WORK/runs/cost_cvae_tau0.5}            # run directory (resumed if it exists)
CONDA_ENV=${CONDA_ENV:-cvae}
MAX_RESUBMIT=${MAX_RESUBMIT:-5}
module load python/3.12-base                       # provides conda on alex
# -------------------------------------------------------------------------------------------------

set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
[ -f train_cost_cvae.py ] || { echo "submit from CVAE_efficient_sampling/ (train_cost_cvae.py not found)"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || { echo "conda env $CONDA_ENV not found"; exit 1; }
echo "job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader), run: $RUN"

srun python train_cost_cvae.py --cache "$CACHE" --out "$RUN" --workers 8 --max_hours 23.7 "$@"
rc=$?
if [ $rc -eq 3 ] && [ "$MAX_RESUBMIT" -gt 0 ]; then
    echo "stopped before the end: resubmitting ($MAX_RESUBMIT left)"
    CACHE="$CACHE" RUN="$RUN" CONDA_ENV="$CONDA_ENV" MAX_RESUBMIT=$((MAX_RESUBMIT - 1)) \
        sbatch --export=ALL slurm_train.sh "$@"
fi
exit $rc
