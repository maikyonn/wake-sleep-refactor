#!/usr/bin/env python3
import os
import subprocess
import glob

# Base directory containing all the data folders
base_dir = "datasets/aria-midi-v1-deduped-ext/data"
# Create a directory for temporary sbatch scripts
temp_sbatch_dir = "temp_sbatch_scripts"
os.makedirs(temp_sbatch_dir, exist_ok=True)

# Get all subdirectories in the base directory
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Create a temporary sbatch script for each folder
for subdir in subdirs:
    # Create results directory if it doesn't exist
    results_dir = f"./real_results_{subdir}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a temporary sbatch script in the temp directory
    temp_script = os.path.join(temp_sbatch_dir, f"temp_sbatch_{subdir}.sh")
    with open(temp_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=classify_{subdir}
#SBATCH -p sched_mit_psfc_gpu_r8      # queue / partition
#SBATCH -N 1                        # 1 node
#SBATCH --ntasks-per-node=1           # one task per node
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out

# ──────────────────────────────────────────────────────────────
# 1.  Activate environment
# ──────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
conda deactivate
source deactivate
which python 
module load deprecated-modules gcc/12.2.0-x86_64         # provides "python" + "torchrun"
module load cuda/12.4.0
source activate /pool001/spangher/wake-sleep-refactor/wake-sleep-env
which python
cd /pool001/spangher/wake-sleep-refactor || exit 1

# ──────────────────────────────────────────────────────────────
# 2.  Pick a master & discover its IP
# ──────────────────────────────────────────────────────────────
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
master_node=${{nodes[0]}}
master_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)

echo "============================================================"
echo "Job      : $SLURM_JOB_ID"
echo "Node     : $master_node   (rank 0 / $SLURM_NNODES)"
echo "Master   : $master_ip:1234"
echo "Start    : $(date)"
echo "============================================================"

# ──────────────────────────────────────────────────────────────
# Enable CUDA_LAUNCH_BLOCKING for debugging CUDA errors
# ──────────────────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=1

# ──────────────────────────────────────────────────────────────
# 3.  Run the evaluation script on the specific folder
# ──────────────────────────────────────────────────────────────
python evaluate_classifier.py --test_dir {os.path.join(base_dir, subdir)} --ckpt_path checkpoints/midi_style/epochepoch=20-valval_loss=0.19.ckpt --mode real --results_dir ./aria-classify-v1/{results_dir}
""")
    
    # Submit the job
    print(f"Submitting job for {subdir}...")
    subprocess.run(["sbatch", temp_script])
    
    # Optionally, remove the temporary script after submission
    # os.remove(temp_script)

print("All jobs submitted!")