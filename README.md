<div align="center">
  <h1>Anchored Diffusion Language Model</h1>

  <a href="https://anchored-diffusion-llm.github.io/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project page badge"></a>
  <a href="https://arxiv.org/pdf/2505.18456"><img src="https://img.shields.io/badge/ArXiv-Preprint-red" alt="ArXiv badge"></a>
  <a href="https://github.com/LituRout/ADLM"><img src="https://img.shields.io/github/stars/LituRout/ADLM?style=social" alt="GitHub stars"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2025-blueviolet" alt="NeurIPS 2025 badge">
</div>

Anchored Diffusion Language Model (ADLM) introduces an alternate noising schedule for masked diffusion language models. Rather than masking tokens uniformly at random, ADLM keeps important *anchor tokens* visible longer during the forward process, and therefore equivalently unmasks them early during denoising. This anchored two-stage framework yields better likelihood estimates and higher-quality text generations. In practice, ADLM:

- recovers up to **25.4%** perplexity improvements on LM1B and OpenWebText over prior DLMs,
- narrows the gap to autoregressive baselines and becomes the first to surpass them in terms of MAUVE score,
- extends gracefully to **seven zero-shot benchmarks** (LAMBADA, PTB, Wikitext-2/103, LM1B, AG News, PubMed, arXiv).

We also derive the Anchored Negative Evidence Lower Bound (ANELBO), establishing theoretical gains in sample complexity and likelihood modeling.

🔔 **News**
- [x] **[2025.10.13]** ADLM pretraining checkpoints are now live.
- [x] **[2025.10.13]** ADLM codebase has been open-sourced.
- [x] **[2025.09.18]** ADLM was accepted to **NeurIPS 2025** 🏆
- [x] **[2025.05.24]** Paper posted on arXiv.


---

## Contents

- [Quickstart ⚡](#quickstart-)
- [Training 🏋️](#training-)
- [Checkpoints 💾](#checkpoints-)
- [Evaluation 🎯](#evaluation-)
  - [Generative metrics 📈](#mauve-score-generative-text-perplexity-gen-ppl-and-entropy)
  - [Zero-shot perplexity 🎯](#zero-shot-perplexity-)
- [Inference notebook 🧪](#inference-notebook-)
- [Baselines 🆚](#baselines-)
- [Acknowledgements 🙏](#acknowledgements-)
- [Citation 📝](#citation-)

---

## Quickstart ⚡

Create the environment and install the FlashAttention dependency:

```bash
conda env create -f requirements.yml
conda activate adlm
pip install flash-attn==2.6.3
```

Set up output directories for checkpoints and logs:

```bash
mkdir -p outputs
mkdir -p watch_folder
mkdir -p ckpts
```

The project configuration lives under [`configs/`](configs). Zero-shot evaluation datasets can be swapped by selecting the appropriate YAML in [`configs/data/`](configs/data/).

---

## Training 🏋️

Launch ADLM training (Slurm job script):

```bash
sbatch scripts/adlm.sh
```

For quick local debugging, you can launch a single-node run with the following command (feel free to trim overrides as needed):

```bash
torchrun \
  adlm_main.py \
  loader.global_batch_size=64 \
  model=small \
  data=openwebtext-split \
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
```


---

## Checkpoints 💾

- Official ADLM checkpoints (medium and large) are provided in our [ADLM Dropbox folder](https://www.dropbox.com/scl/fo/tht8ghwx67cop14pwxanw/AEqLT94IGe7q0RdXc6mPLws?rlkey=le8rp90axip06p2kb6bhgxdam&e=1&dl=0) and also linked on the [project page](https://anchored-diffusion-llm.github.io/). Download and copy them to [`ckpts/`](ckpts) 
- Baseline checkpoints (AR, MDLM, ReMDM) can be downloaded from the MDLM release [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing).

---

## Evaluation 🎯

### MAUVE score, Generative Text Perplexity (Gen PPL) and Entropy 📈

Runs sample generation along with MAUVE, entropy, and generative perplexity metrics:

```bash
sbatch scripts/adlm_eval.sh
```
For quick local debugging without slurm, you can reuse the same evaluation pipeline with a smaller number of sampling steps (128) and small sample batches (10):

```bash
checkpoint_path=ckpts/adlm-medium.ckpt
sampling_steps=128
SEED=5
NUM_SAMPLE_BATCHES=10
generated_seqs_path=outputs/adlm_samples-${NUM_SAMPLE_BATCHES}_seed-${SEED}_steps-${sampling_steps}.json

python -u -m adlm_main \
  mode=sample_eval \
  data=openwebtext-split \
  model=small \
  parameterization=subs \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path="${checkpoint_path}" \
  loader.batch_size=1 \
  loader.eval_batch_size=1 \
  eval.perplexity_batch_size=1 \
  sampling.steps=${sampling_steps} \
  sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
  sampling.generated_seqs_path="${generated_seqs_path}" \
  sampling.sampler=remdm-loop \
  sampling.nucleus_p=0.9 \
  sampling.eta=0.02 \
  sampling.t_on=0.55 \
  sampling.t_off=0.05 \
  sampling.alpha_on=0.9 \
  seed=${SEED} \
  T=0 \
  time_conditioning=false \
  +wandb.offline=true \
  hydra.run.dir="${PWD}/outputs/adlm-${NUM_SAMPLE_BATCHES}-${SEED}"
```
Following standard practice, we evaluate using 5,000 generated samples. If a Slurm time limit interrupts a run with `sampling.num_sample_batches=5000`, rerun with a new seed with fewer batches.

### Zero-shot Perplexity 🎯

Zero-shot perplexity evaluation on Wikitext-2 by default:

```bash
sbatch scripts/adlm_zero_shot_eval.sh
```

**Other zero-shot tasks** (LAMBADA, PTB, Wikitext-103, LM1B, AG News, PubMed, arXiv) are supported via the dataset configs in [`configs/data/`](configs/data/). Update the `data=` override in the script or use Hydra overrides, e.g. `data=ptb`.

---

## Inference notebook 🧪

Generate samples interactively and compute metrics at a smaller scale:

```bash
jupyter notebook notebooks/adlm_inference.ipynb
```

The notebook accepts the same Hydra overrides and checkpoints as the shell scripts.

---

## Baselines 🆚

Submit baseline jobs to reproduce paper results:

```bash
# Autoregressive baseline
sbatch scripts/ar.sh

# Masked Diffusion Language Model
sbatch scripts/mdlm.sh

# Remasking Masked Diffusion Language Model
sbatch scripts/remdm-loop.sh
```

---

## Acknowledgements 🙏

Built on top of [ReMDM](https://github.com/kuleshov-group/remdm), which is developed using [MDLM](https://github.com/kuleshov-group/mdlm) and [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

---

## Citation 📝

```bibtex
@article{rout2025anchored,
  title={Anchored Diffusion Language Model},
  author={Rout, Litu and Caramanis, Constantine and Shakkottai, Sanjay},
  journal={Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
