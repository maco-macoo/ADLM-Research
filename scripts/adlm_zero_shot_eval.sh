#!/bin/bash
# Job script for zero-shot perplexity evaluation for ADLM:
# https://anchored-diffusion-llm.github.io/

#SBATCH -J adlm-zsp                    # Job name
#SBATCH -o watch_folder/%x_%j.out      # log file (out & err)
#SBATCH -p gh                          # Type of gpus needed
#SBATCH -N 1                           # Total number of nodes requested
#SBATCH -t 48:00:00                    # Time limit (hh:mm:ss)
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --open-mode=append             # Do not overwrite logs
#SBATCH -A CGAI24022                   # Allocation/account to charge

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

## Evaluation Config
# Supported checkpoints adlm-medium.ckpt (262B tokens) and adlm-large.ckpt (524B tokens)
# Zero-shot evaluation is available for datasets such as LAMBADA, PTB, Wikitext-2,
# Wikitext-103, LM1B, AG News, PubMed, and arXiv. See `configs/data/` for the
# corresponding Hydra configs.
checkpoint_path=ckpts/adlm-medium.ckpt
T=0
sampling_steps=1024
p=0.9
eta=0.02
t_on=0.55
t_off=0.05
alpha_on=0.9

# Launch evaluation
srun python -u -m adlm_main \
    mode=ppl_eval \
    loader.global_batch_size=16 \
    data=wikitext2 \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/addr-prefix-remdm-loop" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.nucleus_p=${p} \
    sampling.sampler="remdm-loop" \
    sampling.eta=${eta} \
    sampling.t_on=${t_on} \
    sampling.t_off=${t_off} \
    sampling.alpha_on=${alpha_on}