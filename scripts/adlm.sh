#!/bin/bash
# Job script for training ADLM:
# https://anchored-diffusion-llm.github.io/

#SBATCH -J adlm-train                # Job name
#SBATCH -o watch_folder/%x_%j.out    # log file (out & err)
#SBATCH -p gh                        # Type of gpus needed
#SBATCH -N 32                        # Total number of nodes requested
#SBATCH -t 48:00:00                  # Time limit (hh:mm:ss)
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --open-mode=append           # Do not overwrite logs
#SBATCH -A CGAI24022                 # Allocation/account to charge

# Source user shell configuration to ensure module paths are available
source "~/.bashrc"

# Load required toolchains and CUDA libraries
module load gcc/13 cuda/12.4 nvidia_math

# Activate the environment containing project dependencies
conda activate adlm

# Move to the project workspace so relative paths resolve correctly
cd "/work/adlm" # Change this to your project directory

# Configure PyTorch distributed rendezvous endpoints
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=12802

# Emit full stack traces from Hydra on failure for easier debugging
export HYDRA_FULL_ERROR=1

# To enable preemption re-loading, set `hydra.run.dir` or
# `checkpointing.save_dir` explicitly.

# Launch distributed training across SLURM-managed nodes
srun torchrun \
    --nproc-per-node=1 \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    adlm_main.py \
    loader.global_batch_size=512 \
    model=small \
    data=openwebtext-split \
    wandb.name=adlm \
    parameterization=subs \
    model.length=1024 \
    eval.compute_generative_perplexity=True \
    sampling.steps=1000 \
    checkpointing.save_dir="outputs/adlm/" \
    trainer.num_nodes=32 \
    trainer.val_check_interval=10000 \
    trainer.log_every_n_steps=1000 \
    trainer.max_steps=1_000_000 \
    checkpointing.resume_from_ckpt=True \
    time_conditioning=False \
    enable_anchor_loss=True \
    base_scaling_factor1=3e-3 \
    base_scaling_factor2=1 \
    threshold=5