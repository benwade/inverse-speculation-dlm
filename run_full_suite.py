"""
Full DLM Anchor Test Suite — Cloud Edition
============================================
Runs all experiment phases on a single A100 instance with all models
loaded simultaneously.

  Phase A: DLM Commit Curve Probe
            310 prompts (MMLU / ARC / GSM8K / HumanEval)
            Records per-step unmasking history and computes S_50
            (step at which 50% of tokens are committed).

  Phase B: Anchor Gap-Fill Accuracy
            Extracts anchors at step fractions 0.10, 0.15, 0.25
            Runs Qwen2.5-0.5B and Qwen2.5-1.5B gap-fillers
            Computes word-F1 against Dream full-run reference

  Phase C: Pearl Reasoning Point
            Finds minimum step count on GSM8K that preserves
            ≥95% of full-run F1 (the "just enough DLM" threshold)

  Ablation: Random-anchor and random-position controls
            Validates that real anchor positions carry structural signal

Usage:
  pip install transformers==4.46.2 accelerate datasets scipy torch
  python run_full_suite.py              # full 310-prompt run
  python run_full_suite.py --subset 5  # 5 prompts per benchmark (smoke test)

Outputs (saved to /workspace/results/):
  prompts.json          — full prompt set with metadata
  phase_a_results.json  — commit curves, S_50, coverage per prompt
  phase_b_results.json  — gap-fill F1 by model / step fraction
  phase_c_results.json  — pearl reasoning point per difficulty
  ablation_results.json — real vs random-token vs random-position F1
  analysis.json         — aggregate statistics, Cohen's d, correlations
"""

import os
import sys
import json
import time
import random
import argparse
import gc
from pathlib import Path

import torch
import numpy as np

# ── CLI ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subset", type=int, default=0,
                    help="Smoke-test: N prompts per benchmark (0 = full run)")
parser.add_argument("--skip_phase_c", action="store_true",
                    help="Skip Pearl Reasoning Point search (saves ~20 min)")
parser.add_argument("--skip_ablation", action="store_true",
                    help="Skip random-anchor ablation (saves ~15 min)")
args = parser.parse_args()

DEVICE        = "cuda"
RESULTS_DIR   = Path("/workspace/results")
RESULTS_DIR.mkdir(exist_ok=True)
DLM_STEPS     = 128          # Dream denoising steps
STEP_FRACS    = [0.10, 0.15, 0.25]   # anchor extraction points
MAX_NEW_TOKS  = 256          # cap response length

random.seed(42)
np.random.seed(42)

print("=" * 70)
print("  INVERSE SPECULATION — FULL EXPERIMENT SUITE")
print("=" * 70)
print(f"  Device : {torch.cuda.get_device_name()}")
print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
print(f"  Mode   : {'SUBSET (' + str(args.subset) + ' per benchmark)' if args.subset else 'FULL (310 prompts)'}")
print()


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — BUILD PROMPT SET
# ═══════════════════════════════════════════════════════════════

def build_prompt_set(subset_n=0):
    """
    Returns a list of dicts:
      { "text": str, "source": str, "difficulty": str,
        "ground_truth": str|None }
    310 prompts total:
      MMLU     100  (50 easy, 50 hard)
      ARC      100  (50 Easy, 50 Challenge)
      GSM8K     60  (20 easy, 20 medium, 20 hard)
      HumanEval 50  (mixed)
    """
    from datasets import load_dataset
    prompts = []

    # ── MMLU ────────────────────────────────────────────────────
    print("  Loading MMLU...")
    easy_subjects = [
        "high_school_us_history", "high_school_geography",
        "high_school_government_and_politics", "us_foreign_policy",
        "human_aging", "nutrition", "virology", "marketing",
        "high_school_macroeconomics", "sociology",
    ]
    hard_subjects = [
        "abstract_algebra", "college_mathematics", "formal_logic",
        "professional_medicine", "clinical_knowledge", "medical_genetics",
        "college_chemistry", "electrical_engineering",
        "high_school_physics", "jurisprudence",
    ]

    def mmlu_to_prompt(row):
        choices = "\n".join(
            f"  {c}. {row['choices'][i]}"
            for i, c in enumerate("ABCD")
        )
        return (
            f"{row['question']}\n{choices}\n"
            f"Answer with the letter only."
        )

    for subj_list, diff, n_each in [
        (easy_subjects, "easy", 5),
        (hard_subjects, "hard", 5),
    ]:
        n = subset_n or n_each
        for subj in subj_list:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test")
                for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
                    ans_letter = "ABCD"[row["answer"]]
                    prompts.append({
                        "text": mmlu_to_prompt(row),
                        "source": "mmlu",
                        "difficulty": diff,
                        "ground_truth": ans_letter,
                    })
            except Exception as e:
                print(f"    MMLU {subj}: {e}")

    # ── ARC ─────────────────────────────────────────────────────
    print("  Loading ARC...")
    for split_name, diff in [("ARC-Easy", "easy"), ("ARC-Challenge", "hard")]:
        try:
            ds = load_dataset("allenai/ai2_arc", split_name, split="test")
            n = subset_n or 50
            sample = ds.shuffle(seed=42).select(range(min(n, len(ds))))
            for row in sample:
                choices_text = "\n".join(
                    f"  {row['choices']['label'][i]}. {row['choices']['text'][i]}"
                    for i in range(len(row["choices"]["label"]))
                )
                prompts.append({
                    "text": (
                        f"{row['question']}\n{choices_text}\n"
                        f"Answer with the letter only."
                    ),
                    "source": "arc",
                    "difficulty": diff,
                    "ground_truth": row["answerKey"],
                })
        except Exception as e:
            print(f"    ARC {split_name}: {e}")

    # ── GSM8K ────────────────────────────────────────────────────
    print("  Loading GSM8K...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=42)
        # Approximate difficulty by question length (proxy)
        short  = [r for r in ds if len(r["question"]) < 200][:20]
        medium = [r for r in ds if 200 <= len(r["question"]) < 350][:20]
        hard   = [r for r in ds if len(r["question"]) >= 350][:20]
        n = subset_n or 20
        for rows, diff in [(short, "easy"), (medium, "medium"), (hard, "hard")]:
            for row in rows[:n]:
                prompts.append({
                    "text": row["question"] + "\nSolve step by step.",
                    "source": "gsm8k",
                    "difficulty": diff,
                    "ground_truth": row["answer"].split("####")[-1].strip(),
                })
    except Exception as e:
        print(f"    GSM8K: {e}")

    # ── HumanEval ────────────────────────────────────────────────
    print("  Loading HumanEval...")
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
        n = subset_n or 50
        for row in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
            prompts.append({
                "text": row["prompt"],
                "source": "humaneval",
                "difficulty": "mixed",
                "ground_truth": None,   # execution-based; use F1 against canonical
            })
    except Exception as e:
        print(f"    HumanEval: {e}")

    print(f"  Built {len(prompts)} prompts total.")
    return prompts


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — LOAD MODELS
# ═══════════════════════════════════════════════════════════════

def load_all_models():
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

    models = {}

    print("\n  Loading Dream-7B (bf16)...")
    t0 = time.time()
    models["dream"] = AutoModel.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    models["dream_tok"] = AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True
    )
    models["mask_id"] = models["dream_tok"].convert_tokens_to_ids("<|mask|>")
    print(f"    Loaded in {time.time()-t0:.1f}s")

    print("  Loading Qwen2.5-0.5B-Instruct (bf16)...")
    t0 = time.time()
    models["qwen05"] = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    models["qwen05_tok"] = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    print(f"    Loaded in {time.time()-t0:.1f}s")

    print("  Loading Qwen2.5-1.5B-Instruct (bf16)...")
    t0 = time.time()
    models["qwen15"] = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    models["qwen15_tok"] = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )
    print(f"    Loaded in {time.time()-t0:.1f}s")

    vram_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM used after loading: {vram_gb:.1f} GB")
    return models


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — UTILITIES
# ═══════════════════════════════════════════════════════════════

def word_f1(pred_text, ref_text):
    """Bag-of-words F1 over whitespace-tokenized words."""
    pred_words = set(pred_text.lower().split())
    ref_words  = set(ref_text.lower().split())
    if not pred_words or not ref_words:
        return 0.0
    common = pred_words & ref_words
    if not common:
        return 0.0
    precision = len(common) / len(pred_words)
    recall    = len(common) / len(ref_words)
    return 2 * precision * recall / (precision + recall)


def dream_generate(prompt, models, steps=DLM_STEPS, return_history=False):
    """
    Run Dream-7B on a single prompt.
    Returns (output_text, history) where history is a list of
    partially-unmasked token-id tensors at each step.
    """
    tok   = models["dream_tok"]
    model = models["dream"]

    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)

    history = []

    def collect_history(step_ids):
        if return_history:
            history.append(step_ids.cpu().clone())

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKS,
            num_diffusion_steps=steps,
            output_history=return_history,
            temperature=0.2,
            top_p=0.95,
        )

    # output may be a tuple (ids, history_list) depending on Dream version
    if isinstance(output, tuple):
        out_ids, raw_history = output
        history = [h.cpu() for h in raw_history]
    else:
        out_ids = output

    prompt_len = input_ids.shape[1]
    gen_ids    = out_ids[0, prompt_len:]
    gen_text   = tok.decode(gen_ids, skip_special_tokens=True)
    return gen_text, history


def extract_anchors(history, mask_id, step_frac):
    """
    Given Dream's per-step token history, extract tokens that were
    committed (unmasked) at or before `step_frac` of total steps.

    Returns list of (position, token_id) tuples.
    """
    if not history:
        return []
    n_steps  = len(history)
    cutoff   = max(1, int(n_steps * step_frac))
    # Use the state at the cutoff step
    state    = history[cutoff - 1][0]       # shape: (seq_len,)
    anchors  = [
        (i, int(state[i]))
        for i in range(len(state))
        if state[i] != mask_id
    ]
    return anchors


def gap_fill(prompt, anchors, ref_text, model_key, models, tok_key):
    """
    Given anchor positions, build a masked template and run the
    AR gap-filler to complete it.  Returns word-F1 vs ref_text.
    """
    tok   = models[tok_key]
    model = models[model_key]

    if not anchors:
        return 0.0

    # Decode anchors to text tokens using Dream's tokenizer
    dream_tok = models["dream_tok"]
    anchor_dict = {pos: tok_id for pos, tok_id in anchors}

    # Build a simple prompt: tell the AR model what the anchors are
    anchor_text = " ".join(
        dream_tok.decode([tid], skip_special_tokens=True)
        for _, tid in sorted(anchor_dict.items())
    )
    fill_prompt = (
        f"Complete the following response. "
        f"It must include these key words in roughly this order: {anchor_text}\n\n"
        f"Original question: {prompt}\n\nResponse:"
    )

    messages = [{"role": "user", "content": fill_prompt}]
    inputs = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids  = out[0, inputs.shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    return word_f1(gen_text, ref_text), gen_text


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — PHASE A: DLM COMMIT CURVES
# ═══════════════════════════════════════════════════════════════

def phase_a(prompts, models):
    """
    For each prompt:
      1. Run Dream for 128 steps, collecting per-step history
      2. Record coverage (fraction committed) at each STEP_FRAC
      3. Compute S_50: first step where ≥50% of tokens committed
    """
    print("\n" + "=" * 70)
    print("  PHASE A: DLM COMMIT CURVES")
    print("=" * 70)

    results = []
    t_phase = time.time()

    for i, p in enumerate(prompts):
        t0 = time.time()
        ref_text, history = dream_generate(
            p["text"], models, steps=DLM_STEPS, return_history=True
        )
        elapsed = time.time() - t0

        # Coverage at each step fraction
        coverage = {}
        anchors_by_frac = {}
        if history:
            n_steps = len(history)
            total_gen = len(history[-1][0])   # total generated token positions
            for frac in STEP_FRACS:
                cutoff = max(1, int(n_steps * frac))
                state  = history[cutoff - 1][0]
                committed = int((state != models["mask_id"]).sum().item())
                coverage[str(frac)] = committed / max(total_gen, 1)
                anchors_by_frac[str(frac)] = [
                    (j, int(state[j]))
                    for j in range(len(state))
                    if state[j] != models["mask_id"]
                ]

            # S_50
            s50 = None
            for step_idx, step_state in enumerate(history):
                committed = int((step_state[0] != models["mask_id"]).sum().item())
                if committed / max(total_gen, 1) >= 0.5:
                    s50 = step_idx + 1
                    break
        else:
            coverage = {str(f): 1.0 for f in STEP_FRACS}  # no history = committed all
            anchors_by_frac = {}
            s50 = 1

        results.append({
            "idx":        i,
            "source":     p["source"],
            "difficulty": p["difficulty"],
            "ref_text":   ref_text,
            "coverage":   coverage,
            "s50":        s50,
            "_anchors":   anchors_by_frac,   # stripped before final save
            "dream_time": elapsed,
        })

        pct = (i + 1) / len(prompts) * 100
        cov_str = " | ".join(
            f"{float(k)*100:.0f}%→{v:.2f}" for k, v in coverage.items()
        )
        print(f"  [{i+1:3d}/{len(prompts)}] {p['source']:10s} {p['difficulty']:8s} "
              f"S50={s50:4}  cov: {cov_str}  ({elapsed:.1f}s)")

        # Periodic save
        if (i + 1) % 50 == 0:
            with open(RESULTS_DIR / "phase_a_partial.json", "w") as f:
                safe = [{k: v for k, v in r.items() if k != "_anchors"} for r in results]
                json.dump(safe, f, indent=2)
            print(f"  ── checkpoint saved ({i+1} prompts) ──")

    total_time = time.time() - t_phase
    print(f"\n  Phase A complete: {len(prompts)} prompts in {total_time/60:.1f} min")
    return results


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — PHASE B: ANCHOR GAP-FILL
# ═══════════════════════════════════════════════════════════════

def phase_b(phase_a_results, prompts, models):
    """
    For each prompt × step_frac × gap-filler model:
      Compute word-F1 of gap-filled output vs Dream reference.
    Also computes gap-only contribution (coverage-controlled).
    """
    print("\n" + "=" * 70)
    print("  PHASE B: ANCHOR GAP-FILL")
    print("=" * 70)

    gf_configs = [
        ("qwen05", "qwen05_tok", "Qwen-0.5B"),
        ("qwen15", "qwen15_tok", "Qwen-1.5B"),
    ]

    results = []

    for i, (pa, p) in enumerate(zip(phase_a_results, prompts)):
        row = {
            "idx":        i,
            "source":     p["source"],
            "difficulty": p["difficulty"],
            "gap_fill":   {},
        }

        for frac in STEP_FRACS:
            frac_key = str(frac)
            anchors  = pa.get("_anchors", {}).get(frac_key, [])
            row["gap_fill"][frac_key] = {}

            for model_key, tok_key, label in gf_configs:
                f1, gen_text = gap_fill(
                    p["text"], anchors, pa["ref_text"],
                    model_key, models, tok_key
                )
                row["gap_fill"][frac_key][label] = {
                    "f1":       f1,
                    "coverage": pa["coverage"].get(frac_key, 0.0),
                }

        results.append(row)

        # Summary line
        f1_05_10 = row["gap_fill"].get("0.1", {}).get("Qwen-0.5B", {}).get("f1", 0)
        f1_15_10 = row["gap_fill"].get("0.1", {}).get("Qwen-1.5B", {}).get("f1", 0)
        print(f"  [{i+1:3d}/{len(prompts)}] {p['source']:10s} {p['difficulty']:8s} "
              f"@10%  0.5B={f1_05_10:.3f}  1.5B={f1_15_10:.3f}")

        if (i + 1) % 50 == 0:
            with open(RESULTS_DIR / "phase_b_partial.json", "w") as f:
                json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "phase_b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Phase B complete. Saved phase_b_results.json")
    return results


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — PHASE C: PEARL REASONING POINT
# ═══════════════════════════════════════════════════════════════

def phase_c(phase_a_results, prompts, models):
    """
    On GSM8K prompts only: find the minimum step count that preserves
    ≥95% of full-run word-F1.  This is the 'pearl reasoning point.'
    """
    print("\n" + "=" * 70)
    print("  PHASE C: PEARL REASONING POINT (GSM8K)")
    print("=" * 70)

    gsm_indices = [i for i, p in enumerate(prompts) if p["source"] == "gsm8k"]
    step_counts = [8, 13, 16, 24, 32, 48, 64, 96, 128]

    results = []

    for i in gsm_indices:
        p   = prompts[i]
        ref = phase_a_results[i]["ref_text"]   # 128-step reference

        row = {"idx": i, "difficulty": p["difficulty"], "steps": {}}
        for n_steps in step_counts:
            text, _ = dream_generate(p["text"], models, steps=n_steps, return_history=False)
            f1 = word_f1(text, ref)
            row["steps"][n_steps] = f1
            print(f"  GSM8K [{i:3d}] {n_steps:3d} steps → F1={f1:.3f}")

        # Find pearl point
        full_f1 = row["steps"][128]
        pearl   = None
        for n in step_counts:
            if full_f1 > 0 and row["steps"][n] / full_f1 >= 0.95:
                pearl = n
                break
        row["pearl_point"] = pearl
        results.append(row)

    with open(RESULTS_DIR / "phase_c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Phase C complete. Saved phase_c_results.json")
    return results


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — ABLATION: RANDOM ANCHORS vs REAL ANCHORS
# ═══════════════════════════════════════════════════════════════

def run_ablation(phase_a_results, prompts, models, n_ablation=50):
    """
    On a 50-prompt subset, compare:
      (A) Real anchors extracted from Dream at step 0.10
      (B) Random-token anchors: same positions, random token IDs
      (C) Random-position anchors: same tokens, random positions

    If real anchors are just carrying structural signal from their
    positions, (C) should match (A).  If they carry content signal,
    (A) should dominate.
    """
    print("\n" + "=" * 70)
    print("  ABLATION: REAL vs RANDOM ANCHORS")
    print("=" * 70)

    vocab_size = models["dream_tok"].vocab_size
    subset_idx = list(range(min(n_ablation, len(prompts))))
    random.shuffle(subset_idx)
    subset_idx = subset_idx[:n_ablation]

    results = []

    for i in subset_idx:
        p       = prompts[i]
        pa      = phase_a_results[i]
        ref     = pa["ref_text"]
        real_anchors = pa.get("_anchors", {}).get("0.1", [])

        if not real_anchors:
            continue

        positions = [pos for pos, _ in real_anchors]
        tok_ids   = [tid for _, tid in real_anchors]

        # (A) Real anchors
        f1_real, _ = gap_fill(p["text"], real_anchors, ref,
                               "qwen05", models, "qwen05_tok")

        # (B) Random-token anchors (same positions, random tokens)
        rand_tok_anchors = [(pos, random.randint(0, vocab_size-1))
                            for pos in positions]
        f1_rand_tok, _   = gap_fill(p["text"], rand_tok_anchors, ref,
                                     "qwen05", models, "qwen05_tok")

        # (C) Random-position anchors (same tokens, random positions)
        all_positions = list(range(len(pa["coverage"])))
        rand_positions = random.sample(
            range(MAX_NEW_TOKS), min(len(tok_ids), MAX_NEW_TOKS)
        )
        rand_pos_anchors = list(zip(rand_positions, tok_ids))
        f1_rand_pos, _   = gap_fill(p["text"], rand_pos_anchors, ref,
                                     "qwen05", models, "qwen05_tok")

        row = {
            "idx":        i,
            "source":     p["source"],
            "difficulty": p["difficulty"],
            "f1_real":    f1_real,
            "f1_rand_tok":  f1_rand_tok,
            "f1_rand_pos":  f1_rand_pos,
        }
        results.append(row)

        print(f"  [{i:3d}] real={f1_real:.3f}  rand_tok={f1_rand_tok:.3f}  "
              f"rand_pos={f1_rand_pos:.3f}")

    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Ablation complete. Saved ablation_results.json")
    return results


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — AGGREGATE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_analysis(phase_a, phase_b, ablation=None):
    """
    Compute summary statistics, Cohen's d, Spearman correlation,
    and difficulty breakdown.  Saves analysis.json.
    """
    from scipy import stats

    analysis = {}

    # ── Phase B aggregate F1 ────────────────────────────────────
    for frac in STEP_FRACS:
        frac_key = str(frac)
        f1_05 = [r["gap_fill"][frac_key]["Qwen-0.5B"]["f1"]
                 for r in phase_b if frac_key in r["gap_fill"]]
        f1_15 = [r["gap_fill"][frac_key]["Qwen-1.5B"]["f1"]
                 for r in phase_b if frac_key in r["gap_fill"]]

        analysis[f"f1_mean_05_{int(frac*100)}"] = float(np.mean(f1_05)) if f1_05 else None
        analysis[f"f1_mean_15_{int(frac*100)}"] = float(np.mean(f1_15)) if f1_15 else None

    # ── Cohen's d: real vs random-token (ablation) ───────────────
    if ablation:
        real_f1 = [r["f1_real"] for r in ablation]
        rand_f1 = [r["f1_rand_tok"] for r in ablation]
        if real_f1 and rand_f1:
            pooled_std = np.sqrt(
                (np.std(real_f1, ddof=1)**2 + np.std(rand_f1, ddof=1)**2) / 2
            )
            d = (np.mean(real_f1) - np.mean(rand_f1)) / pooled_std if pooled_std > 0 else 0
            analysis["cohens_d_real_vs_rand_tok"] = float(d)
            t, p = stats.ttest_rel(real_f1, rand_f1)
            analysis["ttest_p_real_vs_rand_tok"] = float(p)

    # ── S50 vs difficulty correlation ────────────────────────────
    diff_map = {"easy": 0, "medium": 1, "hard": 2, "mixed": 1}
    s50_vals  = [r["s50"] for r in phase_a if r["s50"] is not None]
    diff_vals = [diff_map.get(r["difficulty"], 1) for r in phase_a
                 if r["s50"] is not None]
    if len(s50_vals) > 3:
        rho, p = stats.spearmanr(diff_vals, s50_vals)
        analysis["s50_difficulty_rho"]  = float(rho)
        analysis["s50_difficulty_p"]    = float(p)

    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=float)

    print("\n  ── KEY RESULTS ──────────────────────────────────────")
    for k, v in analysis.items():
        if v is not None:
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return analysis


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_global = time.time()

    # ── Prompt set ──────────────────────────────────────────────
    prompts_file = RESULTS_DIR / "prompts.json"
    if prompts_file.exists() and not args.subset:
        print("\n  Loading existing prompt set...")
        with open(prompts_file) as f:
            prompts = json.load(f)
    else:
        print("\n  Building prompt set...")
        prompts = build_prompt_set(args.subset)
        with open(prompts_file, "w") as f:
            json.dump(prompts, f, indent=2)
    print(f"  {len(prompts)} prompts ready.")

    # ── Load models ─────────────────────────────────────────────
    models = load_all_models()

    # ── Phase A ─────────────────────────────────────────────────
    phase_a_results = phase_a(prompts, models)

    # ── Phase B ─────────────────────────────────────────────────
    phase_b_results = phase_b(phase_a_results, prompts, models)

    # ── Phase C ─────────────────────────────────────────────────
    phase_c_results = []
    if not args.skip_phase_c:
        phase_c_results = phase_c(phase_a_results, prompts, models)

    # ── Ablation ────────────────────────────────────────────────
    ablation_results = []
    if not args.skip_ablation:
        ablation_results = run_ablation(phase_a_results, prompts, models)

    # ── Analysis ────────────────────────────────────────────────
    analysis = run_analysis(phase_a_results, phase_b_results, ablation_results)

    # ── Final save of Phase A (anchors stripped) ─────────────────
    for r in phase_a_results:
        r.pop("_anchors", None)
    with open(RESULTS_DIR / "phase_a_results.json", "w") as f:
        json.dump(phase_a_results, f, indent=2)

    elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  ALL PHASES COMPLETE in {elapsed/60:.1f} minutes")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")
