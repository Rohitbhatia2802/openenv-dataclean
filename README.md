---
title: OpenEnv Data Clean
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv: Data Cleaning & Validation Environment

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.ai)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-orange)](https://docs.pydantic.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

> A production-grade OpenEnv environment simulating real-world enterprise data cleaning pipelines.
> Submitted to the Meta/HuggingFace OpenEnv Hackathon.

---

## 🎯 Domain Motivation

Data quality is the #1 bottleneck in enterprise AI/ML pipelines. Studies show that data scientists spend **40–80% of their time cleaning data** — a largely repetitive, rule-driven task that is an ideal target for agentic automation.

This environment challenges agents to:
- Identify and quantify data defects from structured observations
- Select the most efficient sequence of cleaning operations
- Balance thoroughness against a finite step budget
- Achieve deterministic, measurable quality thresholds

---

## 📐 API Surface (OpenEnv Spec)

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `reset(task_id: str) → Observation` | Start a new episode |
| `step` | `step(action_dict: dict) → (Observation, Reward, done, info)` | Execute one cleaning action |
| `state` | `state() → State` | Full internal state snapshot |

### HTTP Endpoints (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute action |
| `/state` | GET | Inspect full state |
| `/health` | GET | Service health check |
| `/tasks` | GET | List available tasks |
| `/grader/{id}` | GET | Get grader config for task |
| `/grader/{id}` | POST | Submit episode for grading |
| `/docs` | GET | Swagger UI |

---

## 📊 Observation Space

Each observation is a `Pydantic v2` model with these fields:

```json
{
  "task_id": "fix_missing_price",
  "step_number": 3,
  "max_steps": 20,
  "total_defects": 47,
  "defect_reduction_pct": 61.2,
  "missing_count": 40,
  "sentinel_count": 20,
  "price_below_zero": 20,
  "duplicate_count": 0,
  "phone_format_errors": 0,
  "email_case_errors": 0,
  "age_range_violations": 0,
  "date_order_violations": 0,
  "icd10_format_errors": 0,
  "negative_dosage_count": 0,
  "blank_mandatory_count": 0,
  "cross_column_violations": 0,
  "column_stats": [...],
  "sample_rows": [...],
  "done": false,
  "message": "Imputed 40 missing values using strategy='median'."
}
```

---

## ⚡ Action Space

Agents submit JSON actions from the following operation set:

| Operation | Description | Key Params |
|-----------|-------------|------------|
| `impute_missing` | Fill null values | `strategy`: median\|mean\|mode\|constant |
| `replace_sentinel` | Replace sentinel values | `sentinel`: -1, `strategy`: median |
| `deduplicate` | Remove duplicate rows | `keep`: first\|last |
| `standardize_phone` | Normalize to E.164 | — |
| `lowercase_email` | Force email lowercase | — |
| `clamp_range` | Clamp numeric values | `min_val`, `max_val` |
| `fix_date_order` | Fix DOB > admission | `mode`: swap\|null_dob |
| `fix_icd10` | Fix ICD-10 format | — |
| `fix_negative_dosage` | Abs() negative dosages | — |
| `fill_mandatory` | Fill blank required fields | `value`: string |
| `validate` | No-op grade check | — |
| `export` | Signal episode done | — |

**Example action:**
```json
{
  "operation": "impute_missing",
  "target_columns": ["price"],
  "params": {"strategy": "median"}
}
```

---

## 🎮 Tasks

### Easy — Fix Missing Price Values
- **Dataset**: 200-row retail transaction log
- **Defects**: ~20% null prices + ~10% sentinel -1 prices
- **Target**: Impute all using column median of valid prices
- **Max Steps**: 20 | **Success Threshold**: 0.95

### Medium — Normalize Customer Data Pipeline
- **Dataset**: ~300-row customer CRM export (with ~8% duplicates)
- **Defects**: duplicate rows, mixed-case emails, varied phone formats, age outliers
- **Target**: Deduplicate → lowercase emails → E.164 phones → clamp ages [18,100]
- **Max Steps**: 35 | **Success Threshold**: 0.90

### Hard — Validate Medical Records
- **Dataset**: 400-row hospital EMR export
- **Defects**: 5 types — date-order violations, ICD-10 format errors, negative dosages, blank mandatory fields, cross-column constraint violations
- **Target**: Fix all 5 defect types under a 50-step budget
- **Max Steps**: 50 | **Success Threshold**: 0.85

---

## 🏆 Reward Function

The reward is **dense and step-wise** — every step returns a non-zero signal:

| Component | Value | Trigger |
|-----------|-------|---------|
| Progress reward | +0.00 to +0.30 | Proportional to defects fixed this step |
| Step penalty | -0.01 | Every step (encourages efficiency) |
| Invalid action | -0.05 | Malformed or illegal operation |
| Destructive action | -0.10 | `drop_column` on required columns |
| Completion bonus | +0.20 | First step grade ≥ success_threshold |

- Clamped per step to `[-1.0, 1.0]`; cumulative to `[0.0, 1.0]`
- No sparse 0/1 terminal rewards

---

## 📈 Baseline Agent Scores (GPT-4o, temperature=0)

| Task | Steps | Final Grade | Success |
|------|-------|-------------|---------|
| fix_missing_price | 4 | 0.982 | ✅ |
| normalize_customer_pipeline | 8 | 0.944 | ✅ |
| validate_medical_records | 14 | 0.891 | ✅ |
| **Mean** | **8.7** | **0.939** | **3/3** |

---

## 🚀 Setup & Running

> **Note:** The project already exists locally at `d:\Open_ENV`.
> If you want others to clone it, follow the **Publish to GitHub** steps below first.

### Local (Python) — already on your machine

```bash
# 1. Go to the project folder (already exists)
cd d:\Open_ENV

# 2. Install dependencies (always use --prefer-binary to avoid source builds)
pip install -r requirements.txt --prefer-binary

# 3. Quick test
python test_env.py

# 4. Run demo baseline (no API key needed)
python baseline.py --demo

# 5. Start the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 6. Run GPT-4o baseline (requires API key)
# PowerShell:
$env:OPENAI_API_KEY = "sk-..."
python baseline.py
```

---

### 📤 Publish to GitHub (first time)

```bash
# Step 1 — Initialise git (inside d:\Open_ENV)
cd d:\Open_ENV
git init
git add .
git commit -m "Initial commit: OpenEnv Data Cleaning Environment"

# Step 2 — Create a new repo on GitHub (github.com → New repository)
#           Name it: openenv-dataclean
#           Leave it empty (no README, no .gitignore)

# Step 3 — Add remote and push
git remote add origin https://github.com/<YOUR_USERNAME>/openenv-dataclean.git
git branch -M main
git push -u origin main
```

After pushing, anyone can install with:
```bash
git clone https://github.com/<YOUR_USERNAME>/openenv-dataclean.git
cd openenv-dataclean
pip install -r requirements.txt --prefer-binary
```


### Docker

```bash
# Build
docker build -t openenv-dataclean .

# Run
docker run -p 7860:7860 openenv-dataclean

# Verify
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "fix_missing_price"}'
```

### HuggingFace Spaces

Deploy directly as a Docker Space:
1. Push this repo to HuggingFace
2. Set Space SDK to **Docker**
3. The `Dockerfile` handles the rest — port 7860 is pre-configured

---

## 🧪 Verification Commands

```bash
# Validate OpenEnv spec
openenv validate

# Run baseline
export OPENAI_API_KEY="your_key_here"
python baseline.py

# Build & test Docker
docker build -t openenv-dataclean .
docker run -p 7860:7860 openenv-dataclean &
curl http://localhost:7860/health

# Quick env test
python -c "
from src.env import DataCleaningEnv
e=DataCleaningEnv()
o=e.reset('fix_missing_price')
print('reset OK')
r=e.step({'operation':'validate','target_columns':[],'params':{}})
print('step OK')
print(e.state())
"
```

---

## ⚠️ Limitations

1. **Single-threaded**: The FastAPI server uses 1 worker; the DataCleaningEnv instance is not thread-safe. For concurrent evaluation, run separate processes or add asyncio locks.
2. **In-memory only**: All episode state is ephemeral; server restart clears history.
3. **Phone normalisation**: Only US numbers (10-digit or 1+10-digit) are converted to E.164; international formats are left unchanged.
4. **ICD-10 coverage**: The validation regex checks format only (`[A-Z]\d{2}\.\d{1,2}`); clinical code validity is not checked against the full ICD-10 codebook.
5. **OpenAI baseline cost**: Running all 3 tasks uses approximately 15–40 GPT-4o API calls.

---

## 📁 Project Structure

```
openenv-dataclean/
├── openenv.yaml          # OpenEnv specification
├── server/
│   └── app.py            # FastAPI server
├── baseline.py           # OpenAI baseline agent
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── src/
    ├── __init__.py       # Package init
    ├── models.py         # Pydantic v2 models
    ├── env.py            # DataCleaningEnv (core)
    ├── graders.py        # Deterministic graders
    ├── rewards.py        # Dense reward calculator
    └── tasks/
        ├── __init__.py
        ├── easy.py       # Retail transaction dataset
        ├── medium.py     # Customer CRM dataset
        └── hard.py       # Hospital EMR dataset
```

---

## 📄 License

MIT License © 2024 OpenEnv Hackathon Submission
