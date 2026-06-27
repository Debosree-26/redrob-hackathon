# Redrob Intelligent Candidate Discovery & Ranking
## Team Shankaracharya | POC: Sana Venkata Brahmaiah

---

## Overview

This system ranks 100,000 candidates for a Senior AI Engineer role at Redrob AI. It surfaces the top 100 who genuinely fit — using semantic understanding, behavioral signals, and deep profile analysis rather than keyword matching.

Built around one core principle from the JD: **"10 great matches > 1000 maybes."**

---

## Pipeline Architecture

The full pipeline is two steps:

```
Step 1 — precompute_v2.py   (run once, GPU recommended)
Step 2 — rank_v2_latest.py         (CPU only, < 5 minutes)
```

### Why two steps?

`precompute_v2.py` generates BGE-M3 embeddings for all 100K candidates and the JD — this is compute-heavy (~21 min on A100) but runs only once. `rank_v2_latest.py` loads those precomputed artifacts and ranks in under 1 minute on CPU — well within the 5-minute constraint.

### Sandbox demo

`sandbox_demo.ipynb` is a **single combined notebook** that runs both steps end-to-end on the 50-candidate sample embedded directly in the notebook. No Drive, no uploads, no setup — just click **Runtime → Run all**.

> Note: The sandbox bypasses hard filters because `sample_candidates.json` is a random 50-candidate sample — most fail hard filters by design (they are not pre-filtered). In the full 100K run, ~1,052 candidates survive hard filters. The scoring logic, formula, and penalties are identical.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `precompute_v2.py` | Step 1 — generates BGE-M3 embeddings + metadata for all 100K candidates |
| `rank_v2_latest.py` | Step 2 — loads artifacts, applies hard filters + scoring, outputs top 100 CSV |
| `sandbox_demo.ipynb` | End-to-end demo on 50 embedded sample candidates — no setup needed |
| `redrob_India_runs_v2.ipynb` | Original Colab notebook used to run precompute_v2.py on A100 GPU |
| `rank_local_v2_latest.ipynb` | Local Jupyter notebook used to run ranking step on laptop |
| `submission_v2.csv` | Final ranked output — top 100 candidates |
| `requirements.txt` | Python dependencies |
| `submission_metadata.yaml` | Submission metadata per hackathon spec |

---

## How to Run

### Step 1 — Precompute (run once, GPU recommended)

```bash
pip install FlagEmbedding faiss-cpu numpy pandas tqdm

python precompute_v2.py \
    --input candidates.jsonl \
    --out_dir ./artifacts_v2 \
    --batch_size 256
```

Outputs saved to `artifacts_v2/`:
- `embeddings.npy` — BGE-M3 embeddings for all 100K (~410MB)
- `metadata.pkl` — extracted signals and scoring metadata (~134MB)
- `jd_embedding.npy` — real JD embedding (4KB)

GPU timing: T4 ~2.5 hours | A100 batch_size=512 ~21 minutes

### Step 2 — Rank (< 5 minutes, CPU only)

```bash
python rank_v2_latest.py \
    --artifacts ./artifacts_v2 \
    --out ./submission_v2.csv \
    --top_n 100
```

Output: `submission_v2.csv` — 100 candidates with `candidate_id, rank, score, reasoning`

### Sandbox demo (no setup needed)

Open in Colab and click **Runtime → Run all**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gix0gcNuQvHcLFppjDX0-4OrKzC8v54q)

---

## Architecture

```
candidates.jsonl (487MB)
        │
        ▼ precompute_v2.py (GPU, run once)
        │
        ├── Extract metadata: signals, career flags, avg_tenure,
        │   summary_quality, company_size, FAANG flag
        ├── Build text: FULL summary + FULL career descriptions + skills
        ├── Generate real BGE-M3 JD embedding → jd_embedding.npy
        └── BGE-M3 embeddings → embeddings.npy + metadata.pkl
                │
                ▼ rank_v2_latest.py (CPU, < 5 min)
                │
                ├── Stage 1: Hard filters → ~1,052 survivors from 100K
                ├── Stage 2: Feature (40%) + Semantic (35%) + Behavioral (25%)
                │          × Penalty multipliers (stacked)
                └── Top 100 → submission_v2.csv with reasoning
```

**Key design decision:** `candidates.jsonl` is read exactly once during precompute. The ranking step only loads precomputed artifacts — making the <5 min constraint comfortably achievable.

---

## Approach

### Why not keyword matching?

The JD explicitly warns: *"The right answer is not to find candidates whose skills section contains the most AI keywords. That's a trap."*

We use **BAAI/BGE-M3** — a hybrid dense+sparse retrieval model that:
- Captures semantic meaning ("vector search" ≈ "dense retrieval")
- Handles rare technical keywords (NDCG, Qdrant, FAISS) that pure dense models miss
- Matches the JD's own requirement for hybrid search systems
- Embeds the actual JD text — not a proxy centroid (v2 improvement)

### Stage 1 — Hard Filters (1,052 survivors from 100K)

Rule-based rejection before any ML scoring:

- **Verification:** reject if both email AND phone unverified
- **Availability:** `open_to_work = false` or `notice_period > 60 days`
- **Location:** not willing to relocate AND not in preferred cities
- **Career:** entire career at consulting firms, pure research, CV/speech without NLP
- **Integrity:** experience sum mismatch > 6 months, any skill with zero duration

### Stage 2 — Scoring

**Feature score (40%):** 12 rule-based signals including title relevance, experience range, location fit, avg tenure (title-chaser detection), company size (growth-stage preferred), redrob skill assessments.

**Semantic score (35%):** BGE-M3 embeddings + keyword matching with two multipliers:
- *Shipper multiplier:* rewards candidates who combine ML depth with rapid shipping culture
- *Pre-LLM multiplier:* no penalty for LangChain users who have classical IR roots

**Behavioral score (25%):** Redrob platform signals weighted toward reachability — recruiter response rate (35%), last active date (25%), interview completion, GitHub, LinkedIn.

---

## Key Data Findings

| Finding | Impact on design |
|---------|-----------------|
| Only 16K pass basic verification | Hard filters working correctly |
| Salary max: 74 LPA | >60 LPA = mild misalignment signal |
| Exp mismatch cliff: 6mo → 130+mo | 6-month threshold is data-driven |
| Summary avg 877 chars, all >400 | v1 truncation at 200 chars lost signal; v2 uses full text |
| Full text avg 543 tokens, max 1,161 | No truncation needed — all fit within BGE-M3 2048 limit |

### Honeypot Handling

Two profile integrity checks catch them naturally:
1. Experience sum mismatch > 6 months → 49 candidates
2. Any skill with `duration_months = 0` → 21 candidates

---

## Constraints Met

| Constraint | Status |
|------------|--------|
| Ranking step < 5 min | ✅ < 1 min on CPU |
| ≤ 16GB RAM during ranking | ✅ ~5GB peak |
| CPU only during ranking | ✅ No GPU in rank_v2_latest.py |
| No network during ranking | ✅ All local |
| ≤ 5GB intermediate state | ✅ 410MB + 134MB + 4KB |
| Exactly 100 candidates | ✅ Validated |
| Scores monotonically decreasing | ✅ Validated |
| No duplicate candidate IDs | ✅ Validated |

---

## Requirements

```
numpy>=1.24,<2.0
pandas>=2.0
FlagEmbedding>=1.2
torch>=2.0
transformers>=4.36
faiss-cpu>=1.7
tqdm>=4.65
```

---

## Submission History

| Version | Key changes | Status |
|---------|-------------|--------|
| v1 | Centroid JD proxy, 200-char summary | Submitted |
| v2 | Real JD embedding, full text, shipper/pre-LLM logic, FAANG penalty, Tier-1 city split | Current |

---

## Team

**Team Name:** Shankaracharya
**POC:** Sana Venkata Brahmaiah | brammaayyadsai@gmail.com
**Member:** Debosree Chatterjee | debosree.chatterjee123@gmail.com
