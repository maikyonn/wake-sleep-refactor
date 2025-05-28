#!/bin/bash
#SBATCH --job-name=wake_sleep_ddp
#SBATCH -p sched_mit_psfc_gpu_r8      # queue / partition
#SBATCH -N 1                        # 8 nodes
#SBATCH --ntasks-per-node=1           # one task per node (torchrun will fan-out the GPUs)
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --mem=256GB
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodelist=node[2100-2117]

# ──────────────────────────────────────────────────────────────
# 1.  Activate environment
# ──────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
# Activate spangher bash profile

source /home/spangher/.bashrc
module load miniforge/24.3.0-0
module load cuda/12.4.0
which python

source activate /pool001/spangher/wake-sleep-refactor/wake-sleep-env

# If you normally `conda activate …`, do it here
# conda activate torch
which python
cd /pool001/spangher/wake-sleep-refactor || exit 1

# ──────────────────────────────────────────────────────────────
# 2.  Pick a master & discover its IP
# ──────────────────────────────────────────────────────────────
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
master_node=${nodes[0]}
master_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)

echo "============================================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $master_node   (rank 0 / $SLURM_NNODES)"
echo "Master   : $master_ip:1234"
echo "Start    : $(date)"
echo "============================================================"

# ──────────────────────────────────────────────────────────────
# Enable CUDA_LAUNCH_BLOCKING for debugging CUDA errors
# This makes CUDA operations synchronous, which helps identify
# the exact location of CUDA errors in the code
# ──────────────────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=1

# ──────────────────────────────────────────────────────────────
# 3.  Launch one torchrun per node, one process per GPU
# ──────────────────────────────────────────────────────────────
srun torchrun \
     --nnodes="$SLURM_NNODES" \
     --nproc-per-node="$SLURM_GPUS_PER_NODE" \
     --rdzv-id="$SLURM_JOB_ID" \
     --rdzv-backend=c10d \
     --rdzv-endpoint="$master_ip:1234" \
    train_decoder.py --data_dir cache/aria-deduped-full-pp.pkl --nodes $SLURM_NNODES --num_workers 2 --epochs 100