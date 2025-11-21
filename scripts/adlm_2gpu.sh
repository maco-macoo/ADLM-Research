#!/bin/bash
# ローカル環境用: 2GPUでの訓練スクリプト
# SLURMを使わずに、ローカルマシンの2GPUで実行

# torchrun のデフォルト(29500)が衝突しやすいため、ローカル用のRendezvous設定
# ループバックを明示し、衝突しにくいランダムなポートを選ぶ
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$(shuf -i 20000-39999 -n 1)"

# 環境変数の設定
export HYDRA_FULL_ERROR=1

torchrun \
    --nproc-per-node=2 \
    --nnodes=1 \
    --master_port="${MASTER_PORT}" \
    adlm_main.py \
    loader.global_batch_size=16 \
    model=small \
    data=openwebtext-streaming \
    data.cache_dir=/home/smakoto/PROJECT/datasets \
    wandb.name=adlm \
    parameterization=subs \
    model.length=1024 \
    eval.compute_generative_perplexity=True \
    sampling.steps=1000 \
    checkpointing.save_dir="outputs/adlm/" \
    trainer.num_nodes=1 \
    trainer.val_check_interval=10000 \
    trainer.log_every_n_steps=1000 \
    trainer.max_steps=1_000_000 \
    checkpointing.resume_from_ckpt=True \
    time_conditioning=False \
    enable_anchor_loss=True \
    base_scaling_factor1=3e-3 \
    base_scaling_factor2=1 \
    threshold=5