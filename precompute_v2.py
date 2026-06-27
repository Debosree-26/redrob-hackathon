"""
precompute.py v2 — Redrob Candidate Ranking
Run once (no time limit). Generates embeddings + metadata for all candidates.

v2 changes:
- P1: Full summary saved (not 200-char truncation)
- P2: Full career descriptions included in embedding text
- P3: avg_tenure_months saved in metadata
- P4: summary_quality with specificity signals
- P5: num_companies saved
- P6: current_company_size saved
- P7: Real JD text embedded -> jd_embedding.npy
- P8: Use A100 GPU (batch_size=256 recommended)

Usage:
    python precompute.py --input candidates.jsonl --out_dir ./artifacts --jd job_description.txt
    python precompute.py --input sample_candidates.json --out_dir ./artifacts --jd job_description.txt
"""

import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_BGE = True
except ImportError:
    HAS_BGE = False
    print("⚠️  FlagEmbedding not installed — using dummy embeddings (pilot mode only)")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TODAY       = datetime(2026, 6, 1)
MAX_TOKENS  = 2048

# JD text for embedding — used to generate real jd_embedding.npy
JD_TEXT = """
Senior AI Engineer founding team embeddings dense retrieval semantic search
vector database hybrid search FAISS Qdrant Pinecone Weaviate Milvus Elasticsearch OpenSearch
ranking evaluation NDCG MRR MAP A/B testing offline evaluation online evaluation
Python PyTorch TensorFlow scikit-learn production deployment real users at scale
embedding drift index refresh retrieval quality regression reindexing monitoring
LLM fine-tuning LoRA QLoRA PEFT learning to rank XGBoost neural ranker
HR tech recruiting talent marketplace candidate matching job matching
distributed systems large scale inference open source contributions
hybrid retrieval BM25 sparse dense recommendation system search relevance
ranking system information retrieval NLP natural language processing
sentence transformers BGE E5 cross encoder reranking
evaluation framework offline benchmarks recruiter engagement metrics
production ML systems end to end pipeline owned architected designed
shipped launched deployed real users meaningful scale
"""

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mphasis',
    'hexaware', 'mindtree', 'l&t infotech', 'ltimindtree'
}

FAANG_FIRMS = {
    'google', 'meta', 'microsoft', 'amazon', 'apple', 'netflix'
}

NON_TECH_TITLES = {
    'hr manager', 'human resources', 'mechanical engineer', 'accountant',
    'customer support', 'operations manager', 'content writer',
    'sales executive', 'civil engineer', 'graphic designer',
    'marketing manager', 'sales manager', 'finance manager'
}

REJECT_INDUSTRIES    = {'manufacturing', 'paper products'}
AMBIGUOUS_INDUSTRIES = {'conglomerate', 'transportation', 'media', 'consumer electronics'}

TIER1_CITIES = {
    'noida', 'pune', 'delhi', 'delhi ncr', 'new delhi', 'gurugram', 'gurgaon',
    'hyderabad', 'mumbai', 'bengaluru', 'bangalore', 'chennai', 'kolkata'
}

CV_SPEECH_KEYWORDS = {
    'computer vision', 'object detection', 'image segmentation', 'speech recognition',
    'speech synthesis', 'robotics', 'autonomous', 'lidar', 'image classification'
}

NLP_IR_KEYWORDS = {
    'nlp', 'natural language', 'retrieval', 'search', 'embedding', 'vector',
    'ranking', 'information retrieval', 'text', 'language model', 'transformer'
}

RESEARCH_KEYWORDS = {
    'phd', 'research scientist', 'research engineer', 'academic', 'laboratory',
    'lab', 'published', 'arxiv', 'paper', 'dissertation', 'postdoc'
}

PRODUCTION_KEYWORDS = {
    'production', 'deployed', 'launched', 'shipped', 'real users', 'at scale',
    'serving', 'inference', 'api', 'microservice', 'pipeline'
}

CODING_KEYWORDS = {
    'implemented', 'built', 'developed', 'coded', 'wrote', 'engineered',
    'programmed', 'python', 'sql', 'git', 'github', 'code review'
}

# Specificity signals for summary quality scoring
SPECIFICITY_SIGNALS = {
    '%', 'million', 'billion', '10x', '2x', '3x', 'latency',
    'throughput', 'queries per second', 'qps', 'ms', 'millisecond',
    'users', 'requests', 'accuracy', 'precision', 'recall', 'f1',
    'terabyte', 'petabyte', 'gb', 'tb'
}

# ─── TEXT BUILDER ─────────────────────────────────────────────────────────────
def build_candidate_text(c):
    """
    Build text string for embedding.
    v2: includes full summary + full career descriptions (not truncated).
    Priority: summary → title/headline → skills → career descriptions → certs
    """
    parts = []

    # 1. Full summary (P1: was summary[:200])
    summary = c['profile'].get('summary', '') or ''
    if summary:
        parts.append(summary)

    # 2. Current title + headline
    title   = c['profile'].get('current_title', '')
    headline = c['profile'].get('headline', '')
    if title:
        parts.append(f"Current role: {title}")
    if headline and headline != title:
        parts.append(headline)

    # 3. Skills with proficiency and duration
    skills = c.get('skills', [])
    if skills:
        skill_str = ', '.join(
            f"{s['name']} ({s.get('proficiency','')}, {s.get('duration_months',0)}mo)"
            for s in skills
        )
        parts.append(f"Skills: {skill_str}")

    # 4. Full career descriptions (P2: was only title+company if no desc)
    for job in c.get('career_history', []):
        desc      = job.get('description', '') or ''
        company   = job.get('company', '') or ''
        job_title = job.get('title', '') or ''
        if desc:
            parts.append(f"{job_title} at {company}: {desc}")
        elif company and job_title:
            parts.append(f"{job_title} at {company}")

    # 5. Certifications
    certs = c.get('certifications', [])
    if certs:
        cert_str = ', '.join(str(cert) for cert in certs if cert)
        if cert_str:
            parts.append(f"Certifications: {cert_str}")

    # Join and truncate to MAX_TOKENS (1 token ≈ 4 chars)
    full_text = ' '.join(parts)
    max_chars = MAX_TOKENS * 4
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    return full_text.strip()

# ─── METADATA EXTRACTORS ──────────────────────────────────────────────────────
def is_consulting_firm(company_name):
    name = (company_name or '').lower()
    return any(firm in name for firm in CONSULTING_FIRMS)

def is_faang_firm(company_name):
    name = (company_name or '').lower()
    return any(firm in name for firm in FAANG_FIRMS)

def get_consulting_pct(career_history):
    if not career_history:
        return 0.0
    total = sum(j.get('duration_months', 0) for j in career_history)
    if total == 0:
        return 0.0
    consulting = sum(
        j.get('duration_months', 0) for j in career_history
        if is_consulting_firm(j.get('company', ''))
    )
    return consulting / total

def is_faang_only(career_history):
    """True if entire career is at FAANG/large stable orgs."""
    if not career_history:
        return False
    return all(is_faang_firm(j.get('company', '')) for j in career_history)

def get_last_coding_date(career_history):
    """Return months since last coding evidence."""
    for job in sorted(career_history, key=lambda x: x.get('start_date', ''), reverse=True):
        desc = (job.get('description', '') or '').lower()
        if any(kw in desc for kw in CODING_KEYWORDS):
            end = job.get('end_date')
            if not end or end == 'Present':
                return 0
            try:
                end_dt = datetime.strptime(end[:7], '%Y-%m')
                months_ago = (TODAY.year - end_dt.year) * 12 + (TODAY.month - end_dt.month)
                return months_ago
            except:
                return 0
    return 999

def is_pure_research(career_history, summary):
    text = ' '.join([
        summary or '',
        *[j.get('description', '') or '' for j in career_history],
        *[j.get('title', '') or '' for j in career_history]
    ]).lower()
    has_research   = any(kw in text for kw in RESEARCH_KEYWORDS)
    has_production = any(kw in text for kw in PRODUCTION_KEYWORDS)
    return has_research and not has_production

def is_cv_speech_no_nlp(career_history, summary, skills):
    text = ' '.join([
        summary or '',
        *[j.get('description', '') or '' for j in career_history],
        *[s.get('name', '') for s in skills]
    ]).lower()
    has_cv_speech = any(kw in text for kw in CV_SPEECH_KEYWORDS)
    has_nlp_ir    = any(kw in text for kw in NLP_IR_KEYWORDS)
    return has_cv_speech and not has_nlp_ir

def is_closed_source_no_external(career_history, summary, github_score):
    if github_score != -1:
        return False
    text = ' '.join([
        summary or '',
        *[j.get('description', '') or '' for j in career_history]
    ]).lower()
    external = {'open source', 'github', 'paper', 'published', 'arxiv', 'conference', 'talk', 'blog'}
    has_external = any(kw in text for kw in external)
    total_years  = sum(j.get('duration_months', 0) for j in career_history) / 12
    return total_years >= 5 and not has_external

def get_skill_credibility(skills):
    has_zero  = any(s.get('duration_months', 1) == 0 for s in skills)
    adv_short = sum(
        1 for s in skills
        if s.get('proficiency', '').lower() in ['advanced', 'expert']
        and 0 < s.get('duration_months', 99) < 6
    )
    return has_zero, adv_short

def get_exp_sum_mismatch(career_history, years_of_experience):
    total_months  = sum(j.get('duration_months', 0) for j in career_history)
    claimed_months = years_of_experience * 12
    return abs(total_months - claimed_months)

def get_english_proficiency(languages):
    for lang in languages:
        if lang.get('language', '').lower() == 'english':
            return lang.get('proficiency', '').lower()
    return 'none'

def get_location_bucket(location, country, willing_to_relocate):
    if country and country.lower() not in ('india', 'in'):
        return 'outside_india'
    loc = (location or '').lower()
    if any(city in loc for city in TIER1_CITIES):
        return 'preferred'
    if willing_to_relocate:
        # v2: split into tier1 and tier2 relocation
        # tier1 cities already checked above — if we reach here, it's tier2
        return 'india_tier2_relocate'
    return 'india_no_relocate'

def get_title_bucket(title):
    t = (title or '').lower()
    if any(kw in t for kw in ['hr ', 'human resource', 'mechanical engineer', 'accountant',
                               'customer support', 'operations manager', 'content writer',
                               'sales executive', 'civil engineer', 'graphic designer',
                               'marketing manager']):
        return 'non_tech'
    if any(kw in t for kw in ['business analyst', 'project manager']):
        return 'borderline'
    if any(kw in t for kw in ['senior ai', 'senior ml', 'nlp engineer', 'search engineer',
                               'recommendation', 'applied ml', 'applied scientist',
                               'ml engineer', 'ai engineer', 'machine learning engineer']):
        return 'top'
    if any(kw in t for kw in ['ml', 'machine learning', 'ai ', 'artificial intelligence',
                               'data scientist', 'nlp', 'llm', 'deep learning']):
        return 'strong'
    if any(kw in t for kw in ['data engineer', 'analytics engineer', 'data analyst']):
        return 'adjacent'
    if any(kw in t for kw in ['software engineer', 'backend', 'full stack', 'cloud',
                               'devops', 'frontend', 'java', '.net', 'mobile', 'qa']):
        return 'tech_general'
    if 'computer vision' in t:
        return 'cv_only'
    return 'unknown'

def get_ai_experience_months(career_history, skills):
    """
    v2: only count AI months from product companies (not consulting).
    """
    ai_keywords = {
        'machine learning', 'deep learning', 'nlp', 'ai ', 'artificial intelligence',
        'neural network', 'embedding', 'vector', 'llm', 'transformer', 'bert',
        'pytorch', 'tensorflow', 'sklearn', 'scikit', 'recommendation', 'ranking',
        'retrieval', 'search relevance', 'information retrieval'
    }
    total = 0
    for job in career_history:
        # v2: skip consulting firm months
        if is_consulting_firm(job.get('company', '')):
            continue
        desc  = (job.get('description', '') or '').lower()
        title = (job.get('title', '') or '').lower()
        if any(kw in desc or kw in title for kw in ai_keywords):
            total += job.get('duration_months', 0)

    skill_months = sum(
        s.get('duration_months', 0) for s in skills
        if any(kw in s.get('name', '').lower() for kw in
               ['pytorch', 'tensorflow', 'sklearn', 'nlp', 'bert', 'llm', 'embedding'])
    )
    return max(total, skill_months)

def get_avg_tenure_months(career_history):
    """
    P3: Average months per company — title-chaser detection.
    JD: switching every 1.5 years (18 months) is a red flag.
    """
    durations = [j.get('duration_months', 0) for j in career_history
                 if j.get('duration_months', 0) > 0]
    if not durations:
        return 0
    return sum(durations) / len(durations)

def get_summary_quality(summary):
    """
    P4: Writing ability proxy — word count + specificity signals.
    v2: adds specificity scoring (numbers, metrics, concrete details).
    """
    if not summary:
        return 0.0
    words = len(summary.split())
    text  = summary.lower()

    # Word count score
    if words >= 80:
        word_score = 1.0
    elif words >= 40:
        word_score = 0.7
    elif words >= 15:
        word_score = 0.4
    else:
        word_score = 0.1

    # Specificity score — concrete numbers and metrics
    spec_hits = sum(1 for s in SPECIFICITY_SIGNALS if s in text)
    spec_score = min(spec_hits / 3, 1.0)

    # Combined: 60% word count, 40% specificity
    return 0.6 * word_score + 0.4 * spec_score

def get_industry_flag(current_industry, current_title):
    ind          = (current_industry or '').lower()
    title_bucket = get_title_bucket(current_title)
    if ind in REJECT_INDUSTRIES:
        return 'reject'
    if ind in AMBIGUOUS_INDUSTRIES:
        if title_bucket == 'non_tech':
            return 'reject'
        return 'ok'
    return 'ok'

def get_company_size_score(company_size):
    """
    P6: Growth-stage product companies preferred per JD.
    JD: not a well-scoped Google/Meta role, not pure early-stage chaos.
    """
    size = (company_size or '').lower()
    if '10001' in size:
        return 0.5   # large org — stability seeker risk
    elif '5001' in size:
        return 0.7
    elif '1001' in size:
        return 0.85
    elif '501' in size:
        return 0.9
    elif '201' in size or '51' in size or '11' in size:
        return 1.0   # growth stage — ideal
    elif '1' in size:
        return 0.7   # solo or tiny — possible but no team context
    return 0.8       # unknown — neutral

# ─── MAIN METADATA EXTRACTOR ──────────────────────────────────────────────────
def extract_metadata(c):
    profile  = c['profile']
    signals  = c['redrob_signals']
    career   = c.get('career_history', [])
    skills   = c.get('skills', [])
    languages = c.get('languages', [])

    github_score      = signals.get('github_activity_score', -1)
    summary           = profile.get('summary', '') or ''
    has_zero_skill, adv_short_count = get_skill_credibility(skills)
    exp_mismatch      = get_exp_sum_mismatch(career, profile.get('years_of_experience', 0))

    return {
        # Identity
        'candidate_id': c['candidate_id'],

        # Profile basics
        'current_title':    profile.get('current_title', ''),
        'current_company':  profile.get('current_company', ''),
        'current_industry': profile.get('current_industry', ''),
        'current_company_size': profile.get('current_company_size', ''),  # P6
        'location':         profile.get('location', ''),
        'country':          profile.get('country', ''),
        'years_of_experience': profile.get('years_of_experience', 0),
        'headline':         profile.get('headline', ''),

        # P1: full summary saved (not truncated)
        'full_summary': summary,
        'summary_snippet': summary[:200],   # kept for backward compat

        # Signals
        'profile_completeness_score': signals.get('profile_completeness_score', 0),
        'open_to_work_flag':          signals.get('open_to_work_flag', False),
        'verified_email':             signals.get('verified_email', False),
        'verified_phone':             signals.get('verified_phone', False),
        'linkedin_connected':         signals.get('linkedin_connected', False),
        'notice_period_days':         signals.get('notice_period_days', 999),
        'willing_to_relocate':        signals.get('willing_to_relocate', False),
        'last_active_date':           signals.get('last_active_date', '2020-01-01'),
        'recruiter_response_rate':    signals.get('recruiter_response_rate', 0),
        'interview_completion_rate':  signals.get('interview_completion_rate', 0),
        'offer_acceptance_rate':      signals.get('offer_acceptance_rate', -1),
        'github_activity_score':      github_score,
        'saved_by_recruiters_30d':    signals.get('saved_by_recruiters_30d', 0),
        'salary_max':                 signals.get('expected_salary_range_inr_lpa', {}).get('max', 0),
        'skill_assessment_scores':    signals.get('skill_assessment_scores', {}),
        'preferred_work_mode':        signals.get('preferred_work_mode', ''),

        # Derived flags
        'english_proficiency':    get_english_proficiency(languages),
        'location_bucket':        get_location_bucket(
                                      profile.get('location'), profile.get('country'),
                                      signals.get('willing_to_relocate', False)
                                  ),
        'title_bucket':           get_title_bucket(profile.get('current_title', '')),
        'industry_flag':          get_industry_flag(
                                      profile.get('current_industry'), profile.get('current_title')
                                  ),
        'consulting_pct':         get_consulting_pct(career),
        'is_faang_only':          is_faang_only(career),              # P6: FAANG penalty
        'company_size_score':     get_company_size_score(profile.get('current_company_size', '')),
        'last_coding_months_ago': get_last_coding_date(career),
        'is_pure_research':       is_pure_research(career, summary),
        'is_cv_speech_no_nlp':    is_cv_speech_no_nlp(career, summary, skills),
        'is_closed_source':       is_closed_source_no_external(career, summary, github_score),
        'has_zero_skill_duration': has_zero_skill,
        'adv_short_skill_count':  adv_short_count,
        'exp_sum_mismatch_months': exp_mismatch,
        'ai_experience_months':   get_ai_experience_months(career, skills),  # v2: product cos only
        'avg_tenure_months':      get_avg_tenure_months(career),              # P3
        'summary_quality':        get_summary_quality(summary),               # P4 with specificity
        'num_companies':          len(set(j.get('company', '') for j in career if j.get('company'))),  # P5

        # Raw fields for reasoning + semantic scoring
        'skill_names':       [s['name'] for s in skills[:10]],
        'career_companies':  [j.get('company', '') for j in career[:4]],
        'career_titles':     [j.get('title', '') for j in career[:4]],
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      required=True,          help='candidates.jsonl or sample_candidates.json')
    parser.add_argument('--out_dir',    default='./artifacts',  help='Output directory')
    parser.add_argument('--jd',         default=None,           help='JD text file path (optional)')
    parser.add_argument('--batch_size', type=int, default=256,  help='Embedding batch size (256 for A100, 64 for T4)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Redrob Precompute Pipeline v2")
    print(f"Input     : {args.input}")
    print(f"Output    : {out_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Load candidates ──────────────────────────────────────────────────────
    print("Loading candidates...")
    candidates = []
    if args.input.endswith('.jsonl'):
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        with open(args.input) as f:
            candidates = json.load(f)
    print(f"Loaded {len(candidates):,} candidates\n")

    # ── Extract metadata ─────────────────────────────────────────────────────
    print("Extracting metadata...")
    metadata = []
    for i, c in enumerate(candidates):
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,} / {len(candidates):,}")
        metadata.append(extract_metadata(c))
    print(f"Metadata extracted for {len(metadata):,} candidates\n")

    # ── Build candidate texts ────────────────────────────────────────────────
    print("Building candidate texts (full summary + career descriptions)...")
    texts   = [build_candidate_text(c) for c in candidates]
    avg_len = sum(len(t) for t in texts) / len(texts)
    print(f"Average text length: {avg_len:.0f} chars (~{avg_len/4:.0f} tokens)\n")

    # ── Load BGE-M3 model ────────────────────────────────────────────────────
    if HAS_BGE:
        print("Loading BGE-M3 model...")
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        print("Model loaded\n")
    else:
        print("⚠️  PILOT MODE: using random vectors\n")

    # ── P7: Embed real JD text ───────────────────────────────────────────────
    print("Generating JD embedding (P7: real JD, not centroid proxy)...")
    jd_text = JD_TEXT  # default from config above
    if args.jd and Path(args.jd).exists():
        jd_text = Path(args.jd).read_text()
        print(f"  Using JD from file: {args.jd} ({len(jd_text)} chars)")
    else:
        print(f"  Using built-in JD text ({len(jd_text)} chars)")

    if HAS_BGE:
        jd_result    = model.encode([jd_text], max_length=MAX_TOKENS,
                                    return_dense=True, return_sparse=False,
                                    return_colbert_vecs=False)
        jd_embedding = jd_result['dense_vecs'][0].astype(np.float32)
    else:
        jd_embedding = np.random.randn(1024).astype(np.float32)

    # Normalize
    jd_embedding /= np.linalg.norm(jd_embedding)

    jd_path = out_dir / 'jd_embedding.npy'
    np.save(jd_path, jd_embedding)
    print(f"  jd_embedding.npy saved — shape: {jd_embedding.shape}\n")

    # ── Generate candidate embeddings ────────────────────────────────────────
    print(f"Generating candidate embeddings (batch_size={args.batch_size})...")
    start = datetime.now()

    all_embeddings = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]

        if HAS_BGE:
            result = model.encode(
                batch,
                batch_size=args.batch_size,
                max_length=MAX_TOKENS,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            all_embeddings.append(result['dense_vecs'])
        else:
            vecs  = np.random.randn(len(batch), 1024).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            all_embeddings.append(vecs)

        if (i // args.batch_size) % 10 == 0:
            elapsed = (datetime.now() - start).seconds
            pct     = min((i + args.batch_size) / len(texts) * 100, 100)
            print(f"  {pct:.1f}% — {i + len(batch):,}/{len(texts):,} — {elapsed}s elapsed")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    elapsed    = (datetime.now() - start).seconds
    print(f"\nEmbedding complete — shape: {embeddings.shape} — {elapsed}s total\n")

    # ── Save artifacts ───────────────────────────────────────────────────────
    print("Saving artifacts...")
    emb_path  = out_dir / 'embeddings.npy'
    meta_path = out_dir / 'metadata.pkl'

    np.save(emb_path, embeddings)
    print(f"  embeddings.npy  — {embeddings.nbytes / 1e6:.1f} MB")

    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  metadata.pkl    — {meta_path.stat().st_size / 1e6:.1f} MB")
    print(f"  jd_embedding.npy — {jd_path.stat().st_size / 1e3:.1f} KB")

    print(f"\n{'='*60}")
    print(f"Precompute v2 complete!")
    print(f"  Candidates processed : {len(candidates):,}")
    print(f"  Artifacts saved to   : {out_dir}")
    print(f"  New in v2: full summary, full career text, real JD embedding,")
    print(f"             avg_tenure, summary_quality, company_size, FAANG flag")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
