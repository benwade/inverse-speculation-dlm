"""
Bridge Viability Experiment — Layer Depth vs Anchor Quality
============================================================
Tests whether a large AR model's intermediate layer activations can
serve as high-quality anchor seeds for Dream's denoising process.

  Phase 1: Measure Qwen2.5-32B layer-by-layer confidence
            Taps hidden states at layers 4, 8, 16, 32
            Projects to vocabulary via lm_head
            Records per-token confidence, accuracy, and timing
            Classifies tokens as structural vs content
            Identifies viable layer/threshold configurations

  Phase 2: Inject viable anchors into Dream's starting state
            Uses Dream's generation_tokens_hook_func for injection
            Tests framing-only, content-only, and both anchor sets
            Compares word-F1 against Dream cold-start baseline

EXPERIMENTAL PRINCIPLES:
  1. Injects layer PREDICTIONS, not oracle ground truth
  2. Only tests configurations Phase 1 identifies as viable
  3. Separate easy/hard analysis
  4. Early stops any layer that exceeds 50% of full-model compute

Usage:
  pip install transformers==4.46.2 accelerate bitsandbytes
  python bridge_viability.py

Outputs saved to /workspace/bridge_viability/
"""

import torch
import torch.nn.functional as F
import json
import time
import sys
import gc
from collections import defaultdict
from pathlib import Path

DEVICE   = "cuda"
SAVE_DIR = Path("/workspace/bridge_viability")
SAVE_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────
LAYER_TAPS            = [4, 8, 16, 32]
CONFIDENCE_THRESHOLDS = [0.90, 0.95, 0.99]
DREAM_STEPS           = 128
MAX_NEW_TOKENS        = 256
MAX_COMPUTE_FRACTION  = 0.50   # skip layers that cost >50% of full model

# ── Token classification ─────────────────────────────────────────
FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "to", "for", "on", "with", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "not", "that", "this", "it", "as", "if", "so",
    "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "shall", "must",
}

def classify_token(token_id, tokenizer):
    """Return 'structural' or 'content'."""
    text = tokenizer.decode([token_id]).strip().lower()
    if not text or all(c in '.,!?;:\'"()[]{}' for c in text):
        return "structural"
    if text in FUNCTION_WORDS:
        return "structural"
    return "content"


# ── Prompts ──────────────────────────────────────────────────────
PROMPTS = [
    # Easy — sanity checks
    {"text": "What is the capital of France?",          "difficulty": "easy"},
    {"text": "What does DNA stand for?",                "difficulty": "easy"},
    {"text": "How many sides does a hexagon have?",     "difficulty": "easy"},
    {"text": "What is the boiling point of water in Celsius?", "difficulty": "easy"},
    {"text": "Who wrote Romeo and Juliet?",             "difficulty": "easy"},
    # Medium
    {"text": "Explain the difference between mitosis and meiosis.", "difficulty": "medium"},
    {"text": "What causes tides on Earth?",             "difficulty": "medium"},
    {"text": "Describe how a transformer neural network processes text.", "difficulty": "medium"},
    {"text": "What is the significance of the Krebs cycle?", "difficulty": "medium"},
    {"text": "How does a binary search tree maintain its ordering property?", "difficulty": "medium"},
    # Hard — the real test
    {"text": (
        "A farmer has chickens and cows. He counts 20 heads and 56 legs. "
        "How many chickens does he have?"
     ), "difficulty": "hard"},
    {"text": (
        "You have 8 balls, one slightly heavier. "
        "Using a balance scale only twice, find the heavy ball. "
        "Explain your strategy step by step."
     ), "difficulty": "hard"},
    {"text": (
        "Write a Python function that returns the length of the "
        "longest increasing subsequence of a list of integers."
     ), "difficulty": "hard"},
    {"text": (
        "Explain the causes and consequences of the 2008 financial crisis. "
        "Connect the housing bubble, derivatives, bank failures, "
        "and government response into a coherent narrative."
     ), "difficulty": "hard"},
    {"text": (
        "A patient presents with sudden onset chest pain, diaphoresis, "
        "and jaw pain. Walk through the differential diagnosis, "
        "initial workup, and treatment priorities."
     ), "difficulty": "hard"},
    {"text": (
        "Compare and contrast photosynthesis and cellular respiration. "
        "Explain how they are complementary processes and why both "
        "are necessary for life on Earth."
     ), "difficulty": "hard"},
    {"text": (
        "Prove that the square root of 2 is irrational."
     ), "difficulty": "hard"},
    {"text": (
        "Describe the mechanism of action of ACE inhibitors and explain "
        "why they are preferred over other antihypertensives in "
        "patients with diabetes and chronic kidney disease."
     ), "difficulty": "hard"},
    {"text": (
        "Implement a function to detect a cycle in a linked list "
        "using O(1) space."
     ), "difficulty": "hard"},
    {"text": (
        "Explain how RLHF (Reinforcement Learning from Human Feedback) "
        "is used to align large language models, including the "
        "reward model training and PPO stages."
     ), "difficulty": "hard"},
]


def word_f1(pred, ref):
    pred_w = set(pred.lower().split())
    ref_w  = set(ref.lower().split())
    if not pred_w or not ref_w:
        return 0.0
    common = pred_w & ref_w
    if not common:
        return 0.0
    p = len(common) / len(pred_w)
    r = len(common) / len(ref_w)
    return 2 * p * r / (p + r)


print("=" * 70)
print("  BRIDGE VIABILITY EXPERIMENT")
print("  Layer depth vs anchor quality + Dream injection test")
print("=" * 70)
print(f"  Device : {torch.cuda.get_device_name()}")
print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: Measure 32B layer-by-layer confidence
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PHASE 1: Measuring Qwen2.5-32B layer confidence and timing")
print("=" * 70)

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
)

print("\n  Loading Qwen2.5-32B-Instruct (4-bit)...")
t0 = time.time()

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model_32b = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
).eval()
tokenizer_32b = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True
)
n_layers = model_32b.config.num_hidden_layers
print(f"  32B loaded in {time.time()-t0:.1f}s  |  layers: {n_layers}")
sys.stdout.flush()

# ── Forward hooks to capture hidden states ───────────────────────
captured_states = {}
hook_handles    = []

def make_hook(layer_idx):
    def hook_fn(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured_states[layer_idx] = h.detach().to(torch.float16)
    return hook_fn

for li in LAYER_TAPS:
    if li < n_layers:
        h = model_32b.model.layers[li].register_forward_hook(make_hook(li))
        hook_handles.append(h)

# ── Run Phase 1 ──────────────────────────────────────────────────
phase1_results = []
lm_head = model_32b.lm_head   # project hidden → vocab

for pi, pd in enumerate(PROMPTS):
    prompt     = pd["text"]
    difficulty = pd["difficulty"]
    print(f"\n  [{pi+1:2d}/{len(PROMPTS)}] ({difficulty}) {prompt[:60]}...")
    sys.stdout.flush()

    messages  = [{"role": "user", "content": prompt}]
    input_ids = tokenizer_32b.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)

    # ── Get ground truth via full forward pass ────────────────
    t_full_start = time.time()
    with torch.no_grad():
        full_out = model_32b.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer_32b.eos_token_id,
        )
    t_full = time.time() - t_full_start

    prompt_len   = input_ids.shape[1]
    gt_ids       = full_out[0, prompt_len:].tolist()
    gt_text      = tokenizer_32b.decode(gt_ids, skip_special_tokens=True)
    print(f"    Full 32B ({t_full:.1f}s): {gt_text[:70]}...")

    # Classify each ground-truth token
    gt_types = [classify_token(tid, tokenizer_32b) for tid in gt_ids]

    # ── Measure each layer tap ────────────────────────────────
    layer_results = {}
    for layer_idx in LAYER_TAPS:
        compute_frac = layer_idx / n_layers
        if compute_frac > MAX_COMPUTE_FRACTION:
            print(f"    L{layer_idx}: skipped (compute={compute_frac:.0%} > {MAX_COMPUTE_FRACTION:.0%})")
            continue

        # Re-run just up to this layer by doing a fresh forward pass
        # (hooks already registered; captured_states updated each pass)
        captured_states.clear()
        t_layer_start = time.time()
        with torch.no_grad():
            _ = model_32b(input_ids, output_hidden_states=False)
        t_layer = time.time() - t_layer_start

        if layer_idx not in captured_states:
            print(f"    L{layer_idx}: hook not triggered, skipping")
            continue

        h = captured_states[layer_idx]          # (1, seq_len, hidden)
        # Project last-token hidden state at each generated position
        # We run the prompt through and look at what the layer "thinks"
        # the next tokens should be at each position
        h_gen = h[0, prompt_len - 1:prompt_len - 1 + len(gt_ids)]

        with torch.no_grad():
            logits = lm_head(h_gen.to(lm_head.weight.dtype))   # (n_gen, vocab)
        probs      = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.max(dim=-1)

        top_probs_list = top_probs.cpu().tolist()
        top_ids_list   = top_ids.cpu().tolist()

        for thresh in CONFIDENCE_THRESHOLDS:
            anchors_by_type = {"structural": [], "content": [], "all": []}
            correct_by_type = {"structural": 0, "content": 0, "all": 0}
            total_by_type   = {"structural": 0, "content": 0, "all": 0}

            for pos, (conf, pred_id, gt_id, tok_type) in enumerate(
                zip(top_probs_list, top_ids_list, gt_ids, gt_types)
            ):
                if conf >= thresh:
                    anchors_by_type[tok_type].append((pos, pred_id))
                    anchors_by_type["all"].append((pos, pred_id))
                    if pred_id == gt_id:
                        correct_by_type[tok_type] += 1
                        correct_by_type["all"]    += 1
                    total_by_type[tok_type] += 1
                    total_by_type["all"]    += 1

            for tok_type in ["structural", "content", "all"]:
                n    = total_by_type[tok_type]
                acc  = correct_by_type[tok_type] / n if n > 0 else 0.0
                key  = f"L{layer_idx}_t{thresh}_{tok_type}"
                layer_results[key] = {
                    "layer":             layer_idx,
                    "threshold":         thresh,
                    "token_type":        tok_type,
                    "qualifying_tokens": n,
                    "accuracy":          acc,
                    "compute_fraction":  compute_frac,
                    "layer_time_s":      t_layer,
                    "full_time_s":       t_full,
                    "anchors":           anchors_by_type[tok_type],
                }
                print(f"    L{layer_idx} t{thresh} {tok_type:12s}: "
                      f"n={n:3d}  acc={acc:.3f}  ({t_layer:.1f}s vs {t_full:.1f}s full)")

    phase1_results.append({
        "prompt":        prompt,
        "difficulty":    difficulty,
        "gt_text":       gt_text,
        "gt_ids":        gt_ids,
        "layer_results": layer_results,
    })
    sys.stdout.flush()

# Remove hooks
for h in hook_handles:
    h.remove()
hook_handles.clear()

# Save Phase 1
with open(SAVE_DIR / "phase1_results.json", "w") as f:
    # Don't save full anchor lists (large); save stats only
    save_p1 = []
    for r in phase1_results:
        row = {k: v for k, v in r.items() if k not in ("gt_ids",)}
        row["layer_results"] = {
            k: {sk: sv for sk, sv in v.items() if sk != "anchors"}
            for k, v in r["layer_results"].items()
        }
        save_p1.append(row)
    json.dump(save_p1, f, indent=2)

print("\n  Phase 1 complete. Saved phase1_results.json")

# ── Identify viable configurations ──────────────────────────────
viable_configs = []
for r in phase1_results:
    for key, lr in r["layer_results"].items():
        if lr["qualifying_tokens"] >= 10 and lr["accuracy"] >= 0.90:
            cfg = {
                "layer":       lr["layer"],
                "threshold":   lr["threshold"],
                "token_type":  lr["token_type"],
                "mean_acc":    lr["accuracy"],
                "compute_frac": lr["compute_fraction"],
            }
            if cfg not in viable_configs:
                viable_configs.append(cfg)

viable_configs.sort(key=lambda x: (x["compute_frac"], -x["mean_acc"]))
print(f"\n  Viable configurations ({len(viable_configs)}):")
for vc in viable_configs:
    print(f"    L{vc['layer']} t{vc['threshold']} {vc['token_type']:12s} "
          f"acc={vc['mean_acc']:.3f}  compute={vc['compute_frac']:.0%}")

if not viable_configs:
    print("  No viable configurations found. Exiting before Phase 2.")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════
#  FREE 32B, LOAD DREAM
# ═══════════════════════════════════════════════════════════════

print("\n  Freeing 32B...")
del model_32b
del lm_head
gc.collect()
torch.cuda.empty_cache()
vram_free = torch.cuda.memory_reserved() / 1024**3
print(f"  VRAM reserved after free: {vram_free:.1f} GB")

print("  Loading Dream-7B (bf16)...")
t0 = time.time()
dream_model = AutoModel.from_pretrained(
    "Dream-org/Dream-v0-Instruct-7B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(DEVICE).eval()
dream_tok = AutoTokenizer.from_pretrained(
    "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True
)
mask_id = dream_tok.convert_tokens_to_ids("<|mask|>")
print(f"  Dream loaded in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
#  PHASE 2: Inject anchors into Dream
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PHASE 2: Dream injection test")
print("=" * 70)

def dream_run(prompt, n_steps=DREAM_STEPS, inject_fn=None):
    """
    Run Dream on prompt.  If inject_fn is provided, it is called as
    a generation_tokens_hook_func at step 0 to pre-fill anchors.
    Returns generated text.
    """
    messages  = [{"role": "user", "content": prompt}]
    input_ids = dream_tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)
    prompt_len = input_ids.shape[1]

    kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        num_diffusion_steps=n_steps,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
    )
    if inject_fn is not None:
        kwargs["generation_tokens_hook_func"] = inject_fn

    with torch.no_grad():
        out = dream_model.generate(input_ids, **kwargs)

    if isinstance(out, tuple):
        out = out[0]
    gen_ids  = out[0, prompt_len:]
    gen_text = dream_tok.decode(gen_ids, skip_special_tokens=True)
    return gen_text


def make_inject_fn(anchors_32b, tokenizer_32b, tokenizer_dream):
    """
    Build a hook function that pre-fills Dream's masked sequence with
    anchor tokens translated from the 32B tokenizer space to Dream's.

    Translation is text-based: decode 32B token → re-encode with Dream.
    This introduces subword boundary noise (noted as limitation in paper).
    """
    # Build translated anchor map: {position: dream_token_id}
    translated = {}
    for pos, tid_32b in anchors_32b:
        text_32b = tokenizer_32b.decode([tid_32b], skip_special_tokens=True)
        if not text_32b.strip():
            continue
        ids_dream = tokenizer_dream.encode(text_32b, add_special_tokens=False)
        if ids_dream:
            translated[pos] = ids_dream[0]   # take first subword token

    def hook_fn(step, x, logits):
        """Called by Dream at each step. At step 0, inject anchors."""
        if step == 0:
            for pos, dream_tid in translated.items():
                if pos < x.shape[1]:
                    x[0, pos] = dream_tid
        return x

    return hook_fn, translated


phase2_results = []

for pi, pd in enumerate(PROMPTS):
    prompt     = pd["text"]
    difficulty = pd["difficulty"]
    p1_data    = phase1_results[pi]

    print(f"\n  [{pi+1:2d}/{len(PROMPTS)}] ({difficulty}) {prompt[:60]}...")
    sys.stdout.flush()

    # Cold-start baseline
    t0        = time.time()
    cold_text = dream_run(prompt)
    cold_time = time.time() - t0
    print(f"    Cold ({cold_time:.1f}s): {cold_text[:70]}...")

    row = {
        "prompt":     prompt,
        "difficulty": difficulty,
        "cold_text":  cold_text,
        "cold_time":  cold_time,
        "injections": {},
    }

    # Test each viable configuration
    for vc in viable_configs[:4]:   # cap at 4 configs to control runtime
        key  = f"L{vc['layer']}_t{vc['threshold']}_{vc['token_type']}"
        lr   = p1_data["layer_results"].get(key)
        if lr is None:
            continue

        anchors_32b = lr.get("anchors", [])
        if not anchors_32b:
            continue

        inject_fn, translated = make_inject_fn(
            anchors_32b, tokenizer_32b, dream_tok
        )

        t0          = time.time()
        inject_text = dream_run(prompt, inject_fn=inject_fn)
        inject_time = time.time() - t0

        f1 = word_f1(inject_text, cold_text)   # vs cold baseline
        f1_gt = word_f1(inject_text, p1_data["gt_text"])

        print(f"    {key}: F1_vs_cold={f1:.3f}  F1_vs_32B={f1_gt:.3f}  "
              f"anchors={len(translated)}  ({inject_time:.1f}s)")

        row["injections"][key] = {
            "inject_text":    inject_text,
            "inject_time":    inject_time,
            "n_anchors":      len(translated),
            "f1_vs_cold":     f1,
            "f1_vs_32b_gt":   f1_gt,
        }

    phase2_results.append(row)

with open(SAVE_DIR / "phase2_results.json", "w") as f:
    json.dump(phase2_results, f, indent=2)
print("\n  Phase 2 complete. Saved phase2_results.json")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

for diff in ["easy", "medium", "hard"]:
    rows = [r for r in phase2_results if r["difficulty"] == diff]
    if not rows:
        continue
    cold_f1s = []
    best_inject_f1s = []
    for r in rows:
        if r["injections"]:
            best = max(v["f1_vs_32b_gt"] for v in r["injections"].values())
            cold_f1_vs_gt = word_f1(r["cold_text"], phase1_results[phase2_results.index(r)]["gt_text"])
            cold_f1s.append(cold_f1_vs_gt)
            best_inject_f1s.append(best)

    if cold_f1s:
        import numpy as np
        print(f"\n  {diff.upper()}")
        print(f"    Cold-start F1 vs 32B-GT : {np.mean(cold_f1s):.3f}")
        print(f"    Best-injection F1 vs GT : {np.mean(best_inject_f1s):.3f}")
        delta = np.mean(best_inject_f1s) - np.mean(cold_f1s)
        print(f"    Delta                   : {delta:+.3f}")

print(f"\n  All results in: {SAVE_DIR}")
print("  Done.")
