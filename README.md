# Inverse Speculation: Structural Anchoring from Diffusion Language Models for Edge-Scale Generation

**Ben Wade** · Walden University · [ORCID 0009-0009-5857-7447](https://orcid.org/0009-0009-5857-7447)

---

## Overview

Speculative decoding accelerates inference by having a small model draft tokens that a large model verifies. We invert this paradigm entirely.

Rather than using a small model to speed up a large one, we use a large **Diffusion Language Model (DLM)** to elevate a small one. We exploit a mathematical property of absorbing-state masked diffusion — that token commitments during denoising are permanent and structurally load-bearing — to extract a sparse skeleton of **anchor tokens** from as few as 10% of denoising steps. A sub-billion-parameter autoregressive model then fills the gaps between these anchors, reproducing the DLM's full output at **0.82–0.93 F1** while running entirely on edge hardware.

The DLM never finishes generating. It only needs to *start*.

---

## Key Results

Evaluated on 190 prompts across MMLU, ARC, GSM8K, and HumanEval:

| Step Fraction | Qwen-0.5B F1 | Qwen-1.5B F1 | Mean Coverage |
|:---:|:---:|:---:|:---:|
| 10% (13 steps) | 0.821 | 0.830 | ~30% |
| 15% (19 steps) | 0.887 | 0.896 | ~50% |
| 25% (32 steps) | 0.921 | 0.930 | ~70% |

**Ablation (N=50):** Token identity drives anchor effectiveness, not position.
- Real vs Random-Token: Cohen's d = 6.41, p = 1.21×10⁻¹⁰
- Real vs Random-Position: Cohen's d = 0.10, p = 1.52×10⁻²

**Gap-filler size:** 0.5B matches 1.5B when anchors are provided (Δ = −0.009), confirming the DLM provides the dominant semantic signal.

**Compute reduction:** 128 DLM forward passes → 13, plus ~100 forward passes through a model 14× smaller. The pipeline requires approximately 10–15% of the full DLM's compute budget.

---

## Repository Structure

```
inverse-speculation-dlm/
├── README.md
├── requirements.txt
├── cloud_setup.md          ← RunPod/Vast.ai setup instructions
└── experiments/
    ├── run_full_suite.py   ← Main experiment (Phase A/B/C + ablation)
    └── bridge_viability.py ← Layer-depth vs anchor quality (32B probe)
```

---

## Experiments

### `run_full_suite.py` — Main experiment suite

Runs all phases on a single A100 instance:

- **Phase A:** Dream-7B commit curves on 310 prompts (MMLU / ARC / GSM8K / HumanEval)
- **Phase B:** Qwen-0.5B and Qwen-1.5B gap-fill at step fractions 0.10, 0.15, 0.25
- **Phase C:** Pearl reasoning point — minimum DLM steps preserving ≥95% F1 on GSM8K
- **Ablation:** Real vs random-token vs random-position anchor controls

```bash
# Smoke test (5 prompts per benchmark, ~5 min)
python run_full_suite.py --subset 5

# Full run (310 prompts, ~45 min on A100, ~$1)
python run_full_suite.py
```

Outputs to `/workspace/results/`.

### `bridge_viability.py` — Layer depth experiment

Tests whether Qwen2.5-32B intermediate layer activations can seed Dream's denoising process. Measures per-layer confidence and accuracy, then injects viable anchors into Dream via hook function.

```bash
python bridge_viability.py
```

Outputs to `/workspace/bridge_viability/`.

---

## Setup

See [cloud_setup.md](cloud_setup.md) for full instructions.

Quick start on an A100 instance:

```bash
pip install -r requirements.txt
python run_full_suite.py --subset 5   # smoke test
```

> **Note:** `transformers==4.46.2` is required for Dream-7B compatibility.

---

## Models Used

| Model | Role | VRAM |
|---|---|---|
| [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | DLM anchor source | ~14 GB |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Primary gap-filler | ~1 GB |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Secondary gap-filler | ~3 GB |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | Bridge viability probe | ~20 GB (4-bit) |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{wade2026inverse,
  title   = {Inverse Speculation: Structural Anchoring from Diffusion Language
             Models for Edge-Scale Generation},
  author  = {Wade, Ben},
  year    = {2026},
  orcid   = {0009-0009-5857-7447},
  note    = {Preprint}
}
```

---

## License

MIT
