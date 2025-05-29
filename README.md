<div align="center">
<h1>Anchored Diffusion Language Model</h1>

<a href='https://anchored-diffusion-llm.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2505.18456'><img src='https://img.shields.io/badge/ArXiv-Preprint-red'></a>
[![GitHub](https://img.shields.io/github/stars/LituRout/ADLM?style=social)](https://github.com/LituRout/ADLM)
</div>


Diffusion Language Models (DLMs) promise parallel generation and bidirectional context, yet they underperform autoregressive (AR) models in both <em>likelihood modeling</em> and <em>generated text quality</em>. We identify that this performance gap arises when important tokens (e.g., key words or low-frequency words that anchor a sentence) are masked early in the forward process, limiting contextual information for accurate reconstruction. To address this, we introduce the <em>Anchored Diffusion Language Model (ADLM)</em>, a novel two-stage framework that first predicts distributions over important tokens via an anchor network, and then predicts the likelihoods of missing tokens conditioned on the anchored predictions. ADLM significantly improves test perplexity on LM1B and OpenWebText, achieving up to 25.4% gains over prior DLMs, and narrows the gap with strong AR baselines. It also achieves state-of-the-art performance in zero-shot generalization across seven benchmarks and surpasses AR models in MAUVE score, which marks the first time a DLM generates better human-like text than an AR model. Theoretically, we derive an Anchored Negative Evidence Lower Bound (ANELBO) objective and show that anchoring improves sample complexity and likelihood modeling. Beyond diffusion, anchoring boosts performance in AR models and enhances reasoning in math and logic tasks, outperforming existing chain-of-thought approaches.


![teaser](./data/main-v2.png)


## 🔥 Updates
- **[2025.05.24]** [Paper](https://arxiv.org/pdf/2505.18456) is published on arXiv!


## Citation

```
@article{rout2025anchored,
  title     = {Anchored Diffusion Language Model},
  author    = {Rout, Litu and Caramanis, Constantine and Shakkottai, Sanjay},
  booktitle = {arXiv preprint},
  year      = {2025},
  url       = {https://arxiv.org/pdf/2505.18456}
}
```

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LituRout/ADLM&type=Date)](https://star-history.com/#LituRout/ADLM&Date) -->
