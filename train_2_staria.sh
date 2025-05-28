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
#SBATCH --nodelist=node[2100-2119]


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
    train_staria.py \
    --nodes $SLURM_NNODES \
    --data_dir cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_train.pkl \
    --val_data_dir cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_val.pkl \
    --num_workers 4 \
    --epochs 50 \
    --batch_size 1 \
    --gpus 4 \
    --use_snippet