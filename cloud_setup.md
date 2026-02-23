# Cloud Setup — Inverse Speculation Experiments

## Platform
RunPod, Vast.ai, or Lambda Labs with an **A100-SXM4-80GB** instance.  
Base image: PyTorch 2.5+ with CUDA 12.x (standard on most cloud templates).

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Important:** `transformers==4.46.2` is pinned. Dream-7B will fail
> with newer versions due to `trust_remote_code` changes.

---

## Step 2 — Download models (first run only, ~20 min)

```bash
python3 -c "
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
print('Downloading Dream-7B...')
AutoModel.from_pretrained('Dream-org/Dream-v0-Instruct-7B', trust_remote_code=True)
AutoTokenizer.from_pretrained('Dream-org/Dream-v0-Instruct-7B', trust_remote_code=True)
print('Downloading Qwen-0.5B...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
print('Downloading Qwen-1.5B...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
print('Done.')
"
```

For `bridge_viability.py` only, also download the 32B:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-32B-Instruct',
    quantization_config=bnb, device_map='auto', trust_remote_code=True)
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct', trust_remote_code=True)
print('Done.')
"
```

---

## Step 3 — Create results directory

```bash
mkdir -p /workspace/results
mkdir -p /workspace/bridge_viability
```

---

## Step 4 — Upload scripts

Upload via Jupyter file browser or paste directly. If the file lands in
the wrong directory, find it with:

```bash
find / -name "run_full_suite.py" 2>/dev/null
```

Then run from wherever it landed:

```bash
python /path/to/run_full_suite.py
```

---

## Running the experiments

### Smoke test (5 prompts per benchmark, ~5 min)
```bash
python run_full_suite.py --subset 5
```

### Full run (310 prompts, ~30–45 min on A100, ~$1–2)
```bash
python run_full_suite.py
```

### Skip optional phases to save time
```bash
python run_full_suite.py --skip_phase_c --skip_ablation
```

### Bridge viability experiment (~45 min, requires 32B download)
```bash
python bridge_viability.py
```

---

## Downloading results

Results are saved to `/workspace/results/`. Download via Jupyter file
browser or zip and pull:

```bash
cd /workspace && zip -r results.zip results/
```

Then download `results.zip` through the Jupyter interface.

---

## VRAM budget (A100-SXM4-80GB)

| Model | VRAM |
|---|---|
| Dream-7B (bf16) | ~14 GB |
| Qwen-0.5B (bf16) | ~1 GB |
| Qwen-1.5B (bf16) | ~3 GB |
| All three together | ~18 GB |
| Qwen-32B (4-bit) | ~20 GB |
| 32B + Dream-7B | ~34 GB |

All configurations fit comfortably on an 80 GB A100.

---

## Estimated cost (Vast.ai, A100 ~$1.50/hr)

| Experiment | Time | Cost |
|---|---|---|
| `run_full_suite.py` full run | ~45 min | ~$1.10 |
| `run_full_suite.py --subset 5` | ~5 min | ~$0.12 |
| `bridge_viability.py` | ~45 min | ~$1.10 |
| Both experiments | ~90 min | ~$2.25 |
