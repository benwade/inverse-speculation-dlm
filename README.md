# Inverse Speculation: Structural Anchoring from Diffusion Language Models

This repository accompanies the preprint:

**Inverse Speculation: Structural Anchoring from Diffusion Language Models for Edge-Scale Generation**  
Ben Wade (Independent Researcher)

---

## Overview

Speculative decoding traditionally uses a small model to accelerate a larger model.  
This work inverts that direction:

A Diffusion Language Model (DLM) performs a partial denoising pass (10–25% of steps) and emits committed token anchors.  
A 0.5B autoregressive model then fills the gaps via forced decoding.

Key findings:

- 10% DLM denoising → 0.82 F1 reproduction (N = 190)
- 25% denoising → 0.93 F1 reproduction
- Random-token ablation collapses to 0.002 F1
- Gap-only F1 more than doubles under anchor conditioning (0.475 vs 0.231)
- 0.5B gap-filler matches 1.5B when anchors provided (Δ = −0.009)
- 2.3× sequential speedup
- 89.7% cloud compute reduction

The DLM acts as a structural content architect.  
The AR model performs realization under hard token constraints.

---

## Core Idea

Diffusion language models with absorbing-state masking exhibit a permanence property:

Once a token commits during denoising, it remains fixed.

We exploit early commits as structural anchors.

Instead of compressing text, we compress inference structure.

---

## Pipeline

1. Run Dream-7B for T_partial steps (e.g., 13/128).
2. Extract committed (position, token) pairs.
3. Transmit anchor skeleton.
4. Force-inject anchors into Qwen-0.5B generation loop.
5. Fill gaps autoregressively.

No weight modification required.

---

## Benchmarks

Evaluated on:

- MMLU
- ARC
- GSM8K
- HumanEval

Primary metrics:
- Word-level F1 vs deterministic Dream-7B reference
- Ground-truth accuracy

See paper for full experimental breakdown.

---

## Repository Structure

Initial README
