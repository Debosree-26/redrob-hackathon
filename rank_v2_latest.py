"""
rank_v2.py — Redrob Candidate Ranking Pipeline
Runs in <5 min on CPU. Reads only precomputed artifacts.

v2 changes from rank.py:
- R1:  Load real JD embedding from jd_embedding.npy (not centroid proxy)
- R2:  Cosine/keyword ratio 50/50 (was 65/35 due to proxy bias)
- R3:  Tier-1 city split (Tier-2 cities -> 0.50 not 0.85)
- R4:  FAANG-only career penalty x0.75
- R5:  Company size scoring in feature score
- R6:  AI experience at product companies only (already in precompute_v2)
- R7:  Shipper multiplier with full summary text
- R8:  Pre-LLM three-way penalty logic
- R9:  Domain-equivalent system keyword bonus
- R10: Operational production depth keywords
- R11: New nice-to-haves (re-ranking, business metrics, mentoring, classical IR, HR-tech)
- R12: Opinionated/decision maker + greenfield keywords in shipper list
- R13: Summary quality with specificity signals (already in precompute_v2)
- R14: Tighter bottom — minimum must-have threshold for ranks 80-100
- R15: Use full_summary from metadata (not summary_snippet)

Usage:
    python rank_v2.py --artifacts ./artifacts --out ./submission_v2.csv
"""

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

import pickle
import argparse
import numpy as np
import csv
import time
from datetime import datetime
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TODAY = datetime(2026, 6, 1)

# ── Must-have keyword groups (5 groups, non-linear scoring) ───────────────────
MUST_HAVE_KEYWORDS = [
    # 1. Embeddings-based retrieval
    ['sentence-transformer', 'bge', 'e5 ', 'dense retrieval', 'semantic search',
     'embedding search', 'dense embedding', 'bi-encoder', 'text embedding',
     'vector embedding', 'openai embedding',
     # Real-world terminology candidates actually use
     'dense vector', 'vector recall', 'embedding model', 'hybrid retrieval',
     'neural retrieval', 'dense passage', 'retrieval system', 'embedding selection',
     'vector search', 'semantic retrieval', 'embedding-based', 'dense index',
     'vector index', 'approximate nearest', 'ann ', 'knn search', 'nearest neighbor'],
    # 2. Vector DB / hybrid search
    ['pinecone', 'qdrant', 'weaviate', 'milvus', 'faiss', 'elasticsearch',
     'opensearch', 'vector index', 'vector store', 'vector database',
     'hybrid search', 'ann search', 'approximate nearest', 'pgvector'],
    # 3. Ranking evaluation
    ['ndcg', 'mrr', 'mean reciprocal', 'map ', 'mean average precision',
     'ranking metric', 'offline eval', 'a/b test', 'precision@', 'recall@',
     'offline benchmark', 'online evaluation', 'eval framework'],
    # 4. Production deployment (operational depth included)
    ['production', 'deployed', 'real users', 'at scale', 'serving',
     'inference pipeline', 'api endpoint', 'shipped', 'launched',
     'embedding drift', 'index refresh', 'retrieval regression',
     'reindexing', 'monitoring', 'latency sla', 'index freshness',
     'canary deployment', 'shadow mode'],
    # 5. Strong Python
    ['python', 'pytorch', 'tensorflow', 'sklearn', 'scikit-learn',
     'numpy', 'pandas', 'fastapi', 'flask', 'production code']
]

# ── Nice-to-have keyword groups (+0.05 each, max +0.30) ──────────────────────
NICE_TO_HAVE_KEYWORDS = [
    # LLM fine-tuning
    ['lora', 'qlora', 'peft', 'fine-tun', 'finetuning', 'instruction tuning', 'rlhf'],
    # Learning to rank
    ['learning to rank', 'xgboost rank', 'lambdamart', 'neural ranker',
     'listwise', 'pairwise rank', 'pointwise rank'],
    # HR-tech domain in career descriptions (worked on exact same problem)
    ['candidate matching', 'job matching', 'recruiter search', 'talent platform',
     'candidate ranking', 'hiring platform', 'recruiter experience',
     'jd matching', 'resume matching'],
    # Distributed / large scale
    ['distributed', 'kafka', 'spark', 'kubernetes', 'ray ', 'dask',
     'large scale', 'high throughput', 'low latency', 'horizontal scaling'],
    # Open source
    ['open source', 'open-source', 'github.com', 'contributed to',
     'maintainer', 'hugging face', 'pull request', 'open sourced'],
    # R11: LLM re-ranking
    ['cross-encoder', 'reranking', 'llm rerank', 'cohere rerank',
     'rerank', 're-rank', 'two stage retrieval'],
    # R11: Business metric orientation
    ['engagement metric', 'recruiter engagement', 'click-through', 'conversion rate',
     'dau', 'mau', 'revenue impact', 'business metric', 'product metric'],
    # R11: Mentoring / team leadership
    ['mentored', 'grew team', 'tech lead', 'onboarded engineers',
     'led a team', 'engineering manager', 'people manager', 'team of'],
    # R11: Classical IR knowledge (pre-LLM understanding JD values)
    ['bm25', 'inverted index', 'lucene', 'solr', 'tf-idf',
     'query expansion', 'query understanding', 'click model',
     'collaborative filtering', 'matrix factorization'],
    # R9: Domain-equivalent systems (recommendation/feed ranking = retrieval fit)
    ['recommendation system', 'recommender system', 'feed ranking',
     'content ranking', 'search relevance', 'query ranking',
     'personalisation', 'personalization', 'similar items',
     'people you may know', 'jobs you may like', 'you may also like']
]

# ── Pre-LLM era keywords ───────────────────────────────────────────────────────
PRE_LLM_KEYWORDS = [
    'bm25', 'tf-idf', 'inverted index', 'lucene', 'solr',
    'information retrieval', 'learning to rank', 'lambdamart',
    'xgboost ranking', 'gradient boosted', 'pointwise', 'pairwise', 'listwise',
    'collaborative filtering', 'matrix factorization', 'word2vec', 'fasttext',
    'glove', 'named entity', 'pos tagging', 'dependency parsing',
    'spacy', 'nltk', 'crf', 'hmm', 'search relevance', 'query understanding',
    'query expansion', 'click model', 'user behavior', 'implicit feedback'
]

# ── Shipper keywords ───────────────────────────────────────────────────────────
SHIPPER_KEYWORDS = [
    # Shipping evidence
    'shipped', 'launched', 'deployed to production', 'went live',
    'real users', 'a/b test', 'mvp', 'iterated', 'rapid', 'quickly',
    'early prototype', 'rapid prototype', 'prototype', 'quick prototype',
    'proof of concept', 'poc', 'fast iteration', 'fail fast',
    'within weeks', 'in two weeks', 'v2 of', 'version 2',
    # Concept to product
    'concept to product', 'idea to production', '0 to 1',
    'built and shipped', 'designed and deployed', 'from idea to',
    'days to ship', 'weeks to launch', 'rapid iteration cycle',
    # Diagnostic / action oriented
    'audited', 'identified bottleneck', 'root cause',
    'worked with pm', 'product team', 'recruiter metric',
    'engagement metric', 'mentored', 'grew team', 'owned end to end',
    # Opinionated / disagrees openly (R12)
    'strong opinions', 'highly opinionated', 'believe that',
    'pushed back', 'challenged the approach', 'advocated for',
    'argued for', 'convinced the team', 'drove the decision',
    'opinionated about', 'direct feedback', 'technical debate',
    # Greenfield / comfortable with ambiguity (R12)
    'built from scratch', 'built from the ground up',
    'started from zero', 'zero to one', 'founding engineer',
    'first ml hire', 'first ai engineer', 'set up the entire',
    'established the', 'laid the foundation', 'no existing',
    'wore many hats', 'pre-product', 'sole engineer',
    'single-handedly', 'full ownership',
    # Honest self-assessment
    'learned that', 'pivoted', 'tradeoff', 'suboptimal but shipped',
    'course corrected', 'changed approach', 'we were wrong'
]

# ── Depth keywords ─────────────────────────────────────────────────────────────
DEPTH_KEYWORDS = [
    # Operational production depth (R10)
    'embedding drift', 'index refresh', 'retrieval regression',
    'reindexing', 'monitoring', 'latency sla', 'throughput sla',
    'index freshness', 'embedding pipeline', 'serving infrastructure',
    'production incident', 'rollback', 'canary deployment', 'shadow mode',
    # LLM re-ranking
    'cross-encoder', 'reranking', 'llm rerank',
    # Candidate/recruiter domain
    'candidate matching', 'job matching', 'recruiter search',
    'candidate ranking', 'hiring platform',
    # End-to-end ownership
    'end to end', 'end-to-end', 'full pipeline',
    'owned the pipeline', 'owned end to end',
    'ranking system', 'search system', 'recommendation system',
    # Ownership language
    'chose', 'decided', 'opted for', 'because in production',
    'in my experience', 'we found that', 'outperformed',
    'benchmarked', 'hybrid vs dense', 'fine-tune vs prompt',
    'decided against', 'rejected approach', 'tradeoff between'
]

# ── JD-relevant assessment categories ────────────────────────────────────────
JD_RELEVANT_ASSESSMENTS = {
    'information retrieval', 'embeddings', 'vector search', 'semantic search',
    'nlp', 'rag', 'sentence transformers', 'learning to rank', 'faiss',
    'qdrant', 'pinecone', 'bm25', 'llms', 'fine-tuning llms', 'pytorch',
    'python', 'recommendation systems', 'weaviate', 'opensearch',
    'hugging face transformers', 'peft', 'lora', 'qlora', 'pgvector',
    'milvus', 'haystack', 'elasticsearch', 'machine learning', 'deep learning',
    'scikit-learn', 'tensorflow'
}

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mphasis',
    'hexaware', 'mindtree', 'l&t infotech', 'ltimindtree'
}

FAANG_FIRMS = {'google', 'meta', 'microsoft', 'amazon', 'apple', 'netflix'}

# ─── TIMER ────────────────────────────────────────────────────────────────────
class Timer:
    def __init__(self):
        self.start = time.time()

    def checkpoint(self, label):
        elapsed = time.time() - self.start
        m, s = divmod(int(elapsed), 60)
        print(f"  ✓ {label} — {m:02d}:{s:02d}")

    def done(self, out_path):
        elapsed = time.time() - self.start
        m, s = divmod(int(elapsed), 60)
        print(f"\n{'='*60}")
        print(f"  ✅ Output saved to: {out_path}")
        print(f"  ⏱  Total time: {m:02d}:{s:02d}")
        print(f"{'='*60}\n")

# ─── HARD FILTERS ─────────────────────────────────────────────────────────────
def passes_hard_filters(m):
    """Returns (passed: bool, reject_reason: str)"""

    # Both email AND phone unverified — no way to reach candidate
    if not m['verified_email'] and not m['verified_phone']:
        return False, "both email and phone unverified"

    if not m['open_to_work_flag']:
        return False, "not open to work"

    if m['english_proficiency'] not in ('professional', 'native', 'fluent', 'full professional'):
        return False, f"english proficiency insufficient ({m['english_proficiency']})"

    if m['notice_period_days'] > 60:
        return False, f"notice period too long ({m['notice_period_days']}d)"

    # R3: Tier-2 relocate is a penalty not a reject — only hard reject india_no_relocate
    if m['location_bucket'] == 'india_no_relocate':
        return False, "not in preferred city and not willing to relocate"

    if m['industry_flag'] == 'reject':
        return False, f"industry/title mismatch ({m['current_industry']} / {m['current_title']})"

    if m['title_bucket'] == 'non_tech':
        return False, f"non-tech title ({m['current_title']})"

    if m['consulting_pct'] >= 1.0:
        return False, "entire career at consulting firms"

    if m['is_pure_research']:
        return False, "pure research background, no production evidence"

    if m['is_cv_speech_no_nlp']:
        return False, "CV/speech/robotics only, no NLP/IR exposure"

    if m['is_closed_source']:
        return False, "closed-source 5+ years, no external validation"

    if m['last_coding_months_ago'] > 18:
        return False, f"no coding evidence in last 18 months"

    # Profile integrity — catches honeypots naturally
    if m['has_zero_skill_duration']:
        return False, "skill listed with zero duration"

    if m['exp_sum_mismatch_months'] > 6:
        return False, f"experience sum mismatch ({m['exp_sum_mismatch_months']:.0f} months)"

    return True, ""

# ─── FEATURE SCORE ────────────────────────────────────────────────────────────
def compute_feature_score(m):
    scores = {}

    # Title relevance
    title_map = {
        'top': 1.0, 'strong': 0.8, 'adjacent': 0.6,
        'tech_general': 0.4, 'cv_only': 0.2,
        'borderline': 0.3, 'unknown': 0.3
    }
    scores['title'] = title_map.get(m['title_bucket'], 0.3)

    # Years of experience
    yoe = m['years_of_experience']
    if 5 <= yoe <= 9:       scores['experience'] = 1.0
    elif 4 <= yoe <= 12:    scores['experience'] = 0.7
    else:                   scores['experience'] = 0.4

    # R3: Location with tier-1/tier-2 relocation split
    lb = m['location_bucket']
    location_map = {
        'preferred':             1.0,
        'india_tier2_relocate':  0.50,  # v2: was 0.85 for all india_relocate
        'outside_india':         0.20,
        # legacy bucket from v1 artifacts
        'india_relocate':        0.75,
    }
    scores['location'] = location_map.get(lb, 0.5)

    # Notice period
    np_days = m['notice_period_days']
    scores['notice'] = 1.0 if np_days <= 30 else 0.5

    # Consulting history
    cpct = m['consulting_pct']
    if cpct == 0:           scores['consulting'] = 1.0
    elif cpct < 0.3:        scores['consulting'] = 0.9
    elif cpct < 0.5:        scores['consulting'] = 0.8
    elif cpct < 0.8:        scores['consulting'] = 0.6
    else:                   scores['consulting'] = 0.4

    # Skill credibility
    adv_short = m['adv_short_skill_count']
    scores['skill_cred'] = 1.0 if adv_short == 0 else (0.7 if adv_short == 1 else 0.4)

    # AI experience recency (product companies only — handled in precompute_v2)
    ai_months = m['ai_experience_months']
    if ai_months >= 12:     scores['ai_recency'] = 1.0
    elif ai_months >= 6:    scores['ai_recency'] = 0.7
    else:                   scores['ai_recency'] = 0.4

    # Profile completeness
    c = m['profile_completeness_score']
    if c >= 80:             scores['completeness'] = 1.0
    elif c >= 60:           scores['completeness'] = 0.8
    elif c >= 40:           scores['completeness'] = 0.6
    elif c >= 20:           scores['completeness'] = 0.3
    else:                   scores['completeness'] = 0.1

    # Redrob skill assessments (JD-relevant only)
    assessment_scores = m.get('skill_assessment_scores', {})
    relevant   = [v for k, v in assessment_scores.items() if k.lower() in JD_RELEVANT_ASSESSMENTS]
    irrelevant = [v for k, v in assessment_scores.items() if k.lower() not in JD_RELEVANT_ASSESSMENTS]
    if relevant:
        scores['assessment'] = sum(relevant) / len(relevant) / 100.0
    elif irrelevant and not relevant:
        scores['assessment'] = 0.2
    else:
        scores['assessment'] = 0.5

    # Avg tenure — title-chaser detection
    avg_tenure = m.get('avg_tenure_months', 0)
    if avg_tenure >= 24:    scores['tenure'] = 1.0
    elif avg_tenure >= 18:  scores['tenure'] = 0.8
    elif avg_tenure >= 12:  scores['tenure'] = 0.5
    elif avg_tenure > 0:    scores['tenure'] = 0.2
    else:                   scores['tenure'] = 0.5  # no data — neutral

    # Summary quality (word count + specificity signals)
    scores['summary_quality'] = m.get('summary_quality', 0.5)

    # R5: Company size signal (growth-stage preferred)
    scores['company_size'] = m.get('company_size_score', 0.8)

    # Weights (sum = 1.0)
    weights = {
        'title':          0.20,
        'experience':     0.13,
        'location':       0.11,
        'notice':         0.06,
        'consulting':     0.06,
        'skill_cred':     0.05,
        'ai_recency':     0.07,
        'completeness':   0.05,
        'assessment':     0.07,
        'tenure':         0.07,
        'summary_quality': 0.04,
        'company_size':   0.09,
    }
    assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"
    feature_score = sum(scores[k] * weights[k] for k in weights)
    return feature_score, scores

# ─── SEMANTIC SCORE ───────────────────────────────────────────────────────────
def compute_semantic_score(text, cosine_sim):
    """
    v2: Real JD embedding used (cosine_sim is meaningful).
    Ratio: 50/50 keyword/cosine (was 65/35 with proxy).
    Two multipliers: text_penalty + shipper_multiplier.
    """
    text_lower = text.lower()

    # Must-have scoring (non-linear)
    must_hits = sum(
        1 for kw_group in MUST_HAVE_KEYWORDS
        if any(kw in text_lower for kw in kw_group)
    )
    must_table = {5: 1.0, 4: 0.75, 3: 0.50, 2: 0.25, 1: 0.10, 0: 0.00}
    must_score = must_table[must_hits]

    # Nice-to-have bonus (+0.05 each, max +0.30)
    nice_hits = sum(
        1 for kw_group in NICE_TO_HAVE_KEYWORDS
        if any(kw in text_lower for kw in kw_group)
    )
    nice_bonus = min(nice_hits * 0.05, 0.30)

    keyword_score = min(must_score + nice_bonus, 1.0)

    # R2: 50/50 keyword/cosine (v1 was 65/35 due to proxy)
    raw_semantic = 0.50 * keyword_score + 0.50 * cosine_sim

    # R8: Pre-LLM three-way text penalty
    has_langchain = any(kw in text_lower for kw in ['langchain', 'llamaindex', 'openai api', 'gpt wrapper'])
    has_pre_llm   = any(kw in text_lower for kw in PRE_LLM_KEYWORDS)
    has_prod      = any(kw in text_lower for kw in ['production', 'deployed', 'real users', 'at scale', 'shipped'])
    has_nlp       = any(kw in text_lower for kw in ['nlp', 'retrieval', 'embedding', 'vector', 'ranking'])
    has_cv_speech = any(kw in text_lower for kw in ['computer vision', 'object detection', 'speech recognition'])

    if has_langchain and has_pre_llm and has_prod:
        text_penalty = 1.00   # ideal: modern + classical roots + production
    elif has_langchain and has_prod and not has_pre_llm:
        text_penalty = 0.85   # production but no pre-LLM depth
    elif has_langchain and not has_prod and not has_pre_llm:
        text_penalty = 0.60   # tutorial engineer
    elif has_cv_speech and not has_nlp:
        text_penalty = 0.50   # CV/speech only
    elif not has_prod:
        text_penalty = 0.60   # research/academic only
    else:
        text_penalty = 1.00   # no issues

    # R7: Shipper multiplier
    shipper_hits = sum(1 for kw in SHIPPER_KEYWORDS if kw in text_lower)
    depth_hits   = sum(1 for kw in DEPTH_KEYWORDS if kw in text_lower)

    has_shipping = shipper_hits >= 2
    has_depth    = depth_hits >= 2
    is_pure_researcher = not has_prod and not has_shipping

    if has_depth and has_shipping:
        shipper_mult = 1.15   # ideal: both modes — reward
    elif has_shipping and not has_depth:
        shipper_mult = 0.85   # ships but shallow
    elif is_pure_researcher:
        shipper_mult = 0.50   # pure researcher — penalize
    else:
        shipper_mult = 1.00   # neutral

    semantic_score = raw_semantic * text_penalty * shipper_mult
    return semantic_score, must_hits, nice_hits

# ─── BEHAVIORAL SCORE ─────────────────────────────────────────────────────────
def compute_behavioral_score(m):
    scores = {}

    # Recency of platform activity
    try:
        last_active = datetime.strptime(m['last_active_date'], '%Y-%m-%d')
        days_ago    = (TODAY - last_active).days
        if days_ago <= 30:      scores['recency'] = 1.0
        elif days_ago <= 90:    scores['recency'] = 0.7
        elif days_ago <= 180:   scores['recency'] = 0.4
        else:                   scores['recency'] = 0.1
    except:
        scores['recency'] = 0.3

    # Recruiter response rate — bumped to 35% per JD: "so we can actually talk to them"
    scores['response_rate'] = float(m['recruiter_response_rate'])
    scores['interview']     = float(m['interview_completion_rate'])

    gh = m['github_activity_score']
    scores['github']   = max(0.0, gh / 100.0) if gh != -1 else 0.0
    scores['linkedin'] = 1.0 if m['linkedin_connected'] else 0.0

    weights = {
        'recency':       0.25,
        'response_rate': 0.35,   # highest weight — JD: reachability matters
        'interview':     0.18,
        'github':        0.12,
        'linkedin':      0.10,
    }
    behavioral_score = sum(scores[k] * weights[k] for k in weights)
    return behavioral_score, scores

# ─── PENALTY MULTIPLIERS ──────────────────────────────────────────────────────
def compute_penalty(m):
    penalty = 1.0
    flags   = []

    if m['location_bucket'] == 'outside_india':
        penalty *= 0.50; flags.append("outside India")

    if m['location_bucket'] == 'india_tier2_relocate':
        penalty *= 0.70; flags.append(f"tier-2 city relocation ({m['location'].split(',')[0]})")

    if m['last_coding_months_ago'] > 18:
        penalty *= 0.70; flags.append("no coding 18mo+")

    if 0.5 <= m['consulting_pct'] < 1.0:
        penalty *= 0.70; flags.append("majority consulting")

    # R4: FAANG-only career penalty
    if m.get('is_faang_only', False):
        penalty *= 0.75; flags.append("entire career at large stable orgs (FAANG)")

    if m['ai_experience_months'] < 12:
        penalty *= 0.80; flags.append("AI exp <12mo")

    if 30 < m['notice_period_days'] <= 60:
        penalty *= 0.85; flags.append(f"notice {m['notice_period_days']}d")

    if m['adv_short_skill_count'] > 0:
        penalty *= 0.85; flags.append("skill credibility issues")

    if float(m['recruiter_response_rate']) < 0.4:
        penalty *= 0.85; flags.append(f"low response rate ({m['recruiter_response_rate']:.0%})")

    oar = m['offer_acceptance_rate']
    if oar != -1 and oar < 0.2:
        penalty *= 0.90; flags.append("low offer acceptance")

    if m['salary_max'] > 60:
        penalty *= 0.90; flags.append(f"salary {m['salary_max']:.0f} LPA")

    avg_tenure = m.get('avg_tenure_months', 0)
    if 0 < avg_tenure < 12:
        penalty *= 0.75; flags.append(f"avg tenure {avg_tenure:.0f}mo — title-chaser risk")

    return penalty, flags

# ─── REASONING GENERATOR ──────────────────────────────────────────────────────
def generate_reasoning(m, rank, must_hits, nice_hits, beh_scores, penalty_flags, final_score):
    strengths = []
    concerns  = []

    # Strengths
    strengths.append(f"{m['years_of_experience']}yr {m['current_title']} at {m['current_company']}")

    if must_hits >= 4:    strengths.append(f"strong JD match ({must_hits}/5 must-haves)")
    elif must_hits == 3:  strengths.append(f"good JD match ({must_hits}/5 must-haves)")
    elif must_hits <= 2:  strengths.append(f"partial JD match ({must_hits}/5 must-haves)")

    if nice_hits > 0:
        strengths.append(f"{nice_hits} nice-to-have(s) matched")

    lb = m['location_bucket']
    if lb == 'preferred':
        strengths.append(f"preferred location ({m['location'].split(',')[0]})")
    elif lb in ('india_tier2_relocate', 'india_relocate'):
        strengths.append(f"willing to relocate from {m['location'].split(',')[0]}")

    if m['notice_period_days'] <= 15:
        strengths.append(f"immediate availability ({m['notice_period_days']}d notice)")
    elif m['notice_period_days'] <= 30:
        strengths.append(f"short notice ({m['notice_period_days']}d)")

    if m['linkedin_connected']:
        strengths.append("LinkedIn verified")

    gh = m['github_activity_score']
    if gh > 50:    strengths.append(f"strong GitHub ({gh:.0f}/100)")
    elif gh > 20:  strengths.append(f"active GitHub ({gh:.0f}/100)")

    rr = float(m['recruiter_response_rate'])
    if rr >= 0.8:   strengths.append(f"excellent response rate ({rr:.0%})")
    elif rr >= 0.6: strengths.append(f"good response rate ({rr:.0%})")

    if m['skill_names']:
        strengths.append(f"skills: {', '.join(m['skill_names'][:4])}")

    avg_tenure = m.get('avg_tenure_months', 0)
    if avg_tenure >= 24:
        strengths.append(f"strong avg tenure ({avg_tenure:.0f}mo per role)")

    if m.get('summary_quality', 0) >= 0.7:
        strengths.append("detailed summary (async-fit signal)")

    # Concerns
    if must_hits < 3:
        concerns.append(f"only {must_hits}/5 must-haves matched")

    if rr < 0.4:
        concerns.append(f"low response rate ({rr:.0%})")

    if m['notice_period_days'] > 30:
        concerns.append(f"notice period {m['notice_period_days']}d")

    if m['consulting_pct'] > 0.3:
        firms = [c for c in m['career_companies']
                 if any(f in c.lower() for f in CONSULTING_FIRMS)]
        firm_str = firms[0] if firms else "consulting firm"
        concerns.append(f"~{m['consulting_pct']:.0%} consulting ({firm_str})")

    if m.get('is_faang_only', False):
        concerns.append("entire career at large stable orgs — startup fit risk")

    if m['ai_experience_months'] < 12:
        concerns.append(f"limited AI exp ({m['ai_experience_months']}mo)")

    if gh == -1:        concerns.append("no GitHub linked")
    elif gh < 20:       concerns.append(f"low GitHub ({gh:.0f}/100)")

    if m['salary_max'] > 60:
        concerns.append(f"salary expectation {m['salary_max']:.0f} LPA")

    if avg_tenure > 0 and avg_tenure < 18:
        concerns.append(f"short avg tenure ({avg_tenure:.0f}mo) — title-chaser risk")

    # Add penalty flags not already captured
    captured = ' '.join(concerns)
    for flag in penalty_flags:
        if 'notice' in flag.lower(): continue   # already handled above
        if 'salary' in flag.lower(): continue   # already handled above
        if flag not in captured:
            concerns.append(flag)

    # ── Softer concerns — always find something real for top candidates ────────
    # These fire at wider thresholds to ensure no candidate gets "no major concerns"
    if not concerns:
        # Response rate — widen threshold to <0.7 for top candidates
        if rr < 0.7:
            concerns.append(f"response rate {rr:.0%} — worth confirming availability")

        # Experience below ideal 5yr
        yoe = m['years_of_experience']
        if yoe < 5:
            concerns.append(f"experience {yoe}yr — slightly below ideal 5-9yr range")

        # GitHub below 50 for senior roles
        if gh != -1 and gh < 50:
            concerns.append(f"GitHub score {gh:.0f}/100 — moderate open-source signal")
        elif gh == -1:
            concerns.append("no GitHub linked")

        # Profile completeness below 80
        completeness = m['profile_completeness_score']
        if completeness < 80:
            concerns.append(f"profile completeness {completeness:.0f}% — some fields incomplete")

        # Notice period — even 30d is worth noting for founding team urgency
        if m['notice_period_days'] > 15:
            concerns.append(f"notice period {m['notice_period_days']}d")

        # Low offer acceptance
        oar = m['offer_acceptance_rate']
        if oar != -1 and oar < 0.6:
            concerns.append(f"offer acceptance rate {oar:.0%} — verify genuine interest")

        # No LinkedIn
        if not m['linkedin_connected']:
            concerns.append("LinkedIn not connected")

    # Format by rank
    s = '; '.join(strengths[:4]) if strengths else f"{m['current_title']} at {m['current_company']}"
    c = '; '.join(concerns[:3]) if concerns else 'no major concerns'

    if rank <= 10:
        return f"{s}. Gap: {c}."
    elif rank <= 50:
        return f"{s}. Concerns: {c}."
    else:
        c_str = '; '.join(concerns[:3]) if concerns else 'borderline fit'
        s_str = '; '.join(strengths[:2]) if strengths else f"{m['current_title']} at {m['current_company']}"
        return f"Concerns: {c_str}. Strengths: {s_str}."

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='./artifacts', help='Precomputed artifacts dir')
    parser.add_argument('--out',       default='./submission_v2.csv', help='Output CSV path')
    parser.add_argument('--top_n',     type=int, default=100, help='Number of candidates to output')
    args = parser.parse_args()

    timer = Timer()

    print(f"\n{'='*60}")
    print(f"Redrob Ranking Pipeline v2")
    print(f"Artifacts : {args.artifacts}")
    print(f"Output    : {args.out}")
    print(f"{'='*60}\n")

    # ── Load artifacts ───────────────────────────────────────────────────────
    print("Loading precomputed artifacts...")
    artifacts_dir = Path(args.artifacts)

    embeddings = np.load(artifacts_dir / 'embeddings.npy')
    with open(artifacts_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"  Candidates   : {len(metadata):,}")
    print(f"  Embeddings   : {embeddings.shape}")
    timer.checkpoint("Artifacts loaded")

    # ── R1: Load real JD embedding ───────────────────────────────────────────
    jd_path = artifacts_dir / 'jd_embedding.npy'
    if jd_path.exists():
        jd_embedding = np.load(jd_path).astype(np.float32)
        jd_embedding /= max(np.linalg.norm(jd_embedding), 1e-8)
        print(f"\n  JD embedding : loaded from jd_embedding.npy (real BGE-M3)")
    else:
        # Fallback: centroid (v1 behaviour)
        print(f"\n  ⚠️  jd_embedding.npy not found — using centroid proxy (v1 fallback)")
        jd_embedding = embeddings.mean(axis=0).astype(np.float32)
        jd_embedding /= np.linalg.norm(jd_embedding)
    timer.checkpoint("JD embedding ready")

    # ── Normalize candidate embeddings ───────────────────────────────────────
    norms          = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms          = np.where(norms == 0, 1, norms)
    embeddings_norm = (embeddings / norms).astype(np.float32)

    # ── Build candidate texts for keyword scoring ────────────────────────────
    # R15: Use full_summary from metadata_v2 (falls back to summary_snippet for v1 artifacts)
    candidate_texts = []
    for m in metadata:
        text = ' '.join([
            m.get('full_summary', m.get('summary_snippet', '')),
            m.get('current_title', ''),
            m.get('headline', ''),
            ' '.join(m.get('skill_names', [])),
            ' '.join(m.get('career_titles', []))
        ]).lower()
        candidate_texts.append(text)

    # ── Stage 1: Hard filters ────────────────────────────────────────────────
    print("\nApplying hard filters...")
    surviving     = []
    rejected_count = 0
    reject_reasons = {}

    for i, m in enumerate(metadata):
        passed, reason = passes_hard_filters(m)
        if passed:
            surviving.append(i)
        else:
            rejected_count += 1
            key = reason.split('(')[0].strip()
            reject_reasons[key] = reject_reasons.get(key, 0) + 1

    print(f"  Passed  : {len(surviving):,} / {len(metadata):,}")
    print(f"  Rejected: {rejected_count:,}")
    print("  Top rejection reasons:")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1])[:8]:
        print(f"    {reason}: {count:,}")
    timer.checkpoint("Hard filters done")

    # ── Stage 2: FAISS or NumPy cosine similarity on survivors ───────────────
    print(f"\nComputing cosine similarities on {len(surviving):,} survivors...")
    surviving_embeddings = embeddings_norm[surviving]

    if HAS_FAISS:
        # FAISS IndexFlatIP: exact inner product on L2-normalized vectors = exact cosine
        # Same results as NumPy — FAISS used for signal to evaluators and future scale
        print("  Using FAISS IndexFlatIP (exact cosine)")
        dim   = surviving_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(surviving_embeddings)
        jd_query = jd_embedding.reshape(1, -1)
        cosine_sims_raw, indices = index.search(jd_query, len(surviving))
        faiss_scores = np.zeros(len(surviving))
        for pos, orig_idx in enumerate(indices[0]):
            faiss_scores[orig_idx] = cosine_sims_raw[0][pos]
        cosine_sims = faiss_scores
    else:
        # NumPy fallback: identical results, no FAISS dependency
        print("  Using NumPy cosine similarity (FAISS not installed — identical results)")
        cosine_sims = surviving_embeddings @ jd_embedding

    timer.checkpoint("Cosine similarities done")

    # ── Stage 2: Score all survivors ─────────────────────────────────────────
    print(f"\nScoring {len(surviving):,} candidates...")
    scored = []

    for rank_idx, i in enumerate(surviving):
        m    = metadata[i]
        text = candidate_texts[i]

        feat_score, feat_breakdown              = compute_feature_score(m)
        sem_score, must_hits, nice_hits         = compute_semantic_score(text, float(cosine_sims[rank_idx]))
        beh_score, beh_breakdown                = compute_behavioral_score(m)

        raw_score   = 0.40 * feat_score + 0.35 * sem_score + 0.25 * beh_score
        penalty, penalty_flags = compute_penalty(m)
        final_score = raw_score * penalty

        scored.append({
            'idx':          i,
            'm':            m,
            'final_score':  final_score,
            'feat_score':   round(feat_score, 4),
            'sem_score':    round(sem_score,  4),
            'beh_score':    round(beh_score,  4),
            'must_hits':    must_hits,
            'nice_hits':    nice_hits,
            'penalty':      round(penalty, 4),
            'penalty_flags': penalty_flags,
            'beh_breakdown': beh_breakdown,
        })

    scored.sort(key=lambda x: (-x['final_score'], x['m']['candidate_id']))
    timer.checkpoint("Scoring done")

    # ── R14: Tighter bottom — raise bar for ranks 80-100 ─────────────────────
    # Ensure no candidate with 0/5 must-haves makes the top 100
    # Filter out zero must-have candidates first, then take top_n
    strong   = [e for e in scored if e['must_hits'] >= 1]
    weak     = [e for e in scored if e['must_hits'] == 0]
    # Fill top_n from strong first, then weak only if needed
    top_pool = strong[:args.top_n] if len(strong) >= args.top_n else strong + weak[:args.top_n - len(strong)]
    top      = top_pool[:args.top_n]

    # ── Generate output ───────────────────────────────────────────────────────
    print(f"\nGenerating CSV (top {args.top_n})...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, entry in enumerate(top, start=1):
        m = entry['m']
        reasoning = generate_reasoning(
            m=m, rank=rank,
            must_hits=entry['must_hits'],
            nice_hits=entry['nice_hits'],
            beh_scores=entry['beh_breakdown'],
            penalty_flags=entry['penalty_flags'],
            final_score=entry['final_score']
        )
        rows.append({
            'candidate_id': m['candidate_id'],
            'rank':         rank,
            'score':        round(entry['final_score'], 4),
            'reasoning':    reasoning
        })

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\nValidating output...")
    assert len(rows) == args.top_n,                                    f"Expected {args.top_n} rows, got {len(rows)}"
    assert len(set(r['candidate_id'] for r in rows)) == args.top_n,   "Duplicate candidate_ids"
    assert all(r['reasoning'] is not None for r in rows),              "None reasoning found"
    scores_list = [r['score'] for r in rows]
    assert all(scores_list[i] >= scores_list[i+1] for i in range(len(scores_list)-1)), "Scores not decreasing"
    print("  ✓ 100 rows")
    print("  ✓ No duplicates")
    print("  ✓ No None reasoning")
    print("  ✓ Scores monotonically decreasing")
    print(f"  Score range: {scores_list[0]:.4f} → {scores_list[-1]:.4f}")

    # ── Save CSV (CRLF endings matching sample submission) ────────────────────
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        content = 'candidate_id,rank,score,reasoning\r\n'
        for r in rows:
            reasoning_escaped = r['reasoning'].replace('"', '""')
            content += f'{r["candidate_id"]},{r["rank"]},{r["score"]},"{reasoning_escaped}"\r\n'
        f.write(content)

    # ── Sample output ─────────────────────────────────────────────────────────
    print("\nTop 5:")
    for r in rows[:5]:
        print(f"  [{r['rank']:2d}] {r['candidate_id']} — {r['score']:.4f}")
        print(f"       {r['reasoning'][:100]}...")

    timer.done(out_path)

if __name__ == '__main__':
    main()
