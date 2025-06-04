#!/bin/bash
#SBATCH --job-name=wake_sleep_ddp
#SBATCH -p sched_mit_psfc_gpu_r8      # queue / partition
#SBATCH --nodes=5               # This needs to match Fabric(num_nodes=1)
#SBATCH --ntasks-per-node=4     # This needs to match Fabric(devices=4)
#SBATCH --gpus-per-node=4 
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --nodelist=node[2100-2110]

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
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


# ──────────────────────────────────────────────────────────────
# 3.  Launch one torchrun per node, one process per GPU
# ──────────────────────────────────────────────────────────────
srun python fabric_xt_2.py \
    --train_pkl_file cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_train.pkl \
    --val_pkl_file cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_val.pkl \
    --epochs 4 \
    --lr 1e-4 \
    --devices 4 \
    --num_nodes $SLURM_NNODES \
    --batch_size 1 \
    --num_workers 4 \
    --checkpoint_dir checkpoints/staria_xt_2