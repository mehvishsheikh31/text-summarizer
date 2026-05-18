import streamlit as st
import re
import math
from collections import defaultdict

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextRank Summarizer",
    page_icon="✦",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0c0c0c;
    color: #e8e4dc;
    font-family: 'Manrope', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(200,169,110,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 90%, rgba(100,160,130,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(255,255,255,0.01) 0%, transparent 80%),
        #0c0c0c;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 800px; padding: 2.5rem 2rem 5rem; }

/* ── Title ── */
.title-block {
    text-align: center;
    margin-bottom: 3rem;
    padding-top: 1.5rem;
}
.title-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #c8a96e;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    border: 1px solid rgba(200,169,110,0.3);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    margin-bottom: 1rem;
}
.title-block h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    font-weight: 400;
    color: #e8e4dc;
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem;
    line-height: 1.05;
}
.title-block h1 em {
    font-style: italic;
    color: #c8a96e;
}
.title-block .subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #444;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0.8rem 0 0;
}
.divider {
    width: 1px;
    height: 32px;
    background: linear-gradient(to bottom, transparent, #c8a96e, transparent);
    margin: 1rem auto;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #444;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e1e1e;
}

/* ── Input area ── */
.stTextArea textarea {
    background: #131313 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    padding: 1.1rem 1.2rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(200,169,110,0.5) !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.07) !important;
}
.stTextArea textarea::placeholder { color: #333 !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0.4rem 0; }
label[data-testid="stWidgetLabel"] {
    color: #555 !important;
    font-size: 0.72rem !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
.stButton button {
    background: linear-gradient(135deg, #c8a96e 0%, #b8924a 100%) !important;
    color: #0c0c0c !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(200,169,110,0.15) !important;
}
.stButton button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton button:active { transform: translateY(0) !important; }

/* ── Output section ── */
.output-wrapper {
    margin-top: 2rem;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.summary-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.8rem;
}
.summary-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #c8a96e;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.sentence-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #555;
    background: #161616;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 0.2rem 0.7rem;
    letter-spacing: 0.08em;
}

.summary-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #c8a96e;
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    font-size: 0.97rem;
    line-height: 1.9;
    color: #d8d3c8;
    font-family: 'Manrope', sans-serif;
    position: relative;
}
.summary-card .sent {
    display: inline;
}
.summary-card .sent + .sent::before {
    content: ' ';
}
.sent-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #c8a96e;
    opacity: 0.6;
    vertical-align: super;
    margin-right: 2px;
}

/* ── Stats ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 1rem;
}
.stat-box {
    background: #0e0e0e;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    padding: 1rem 0.8rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-box:hover { border-color: rgba(200,169,110,0.2); }
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: #c8a96e;
    display: block;
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: #3a3a3a;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    display: block;
    margin-top: 0.35rem;
}

/* ── How it works ── */
.how-it-works {
    margin-top: 3rem;
    border-top: 1px solid #161616;
    padding-top: 2rem;
}
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.step-card {
    background: #0e0e0e;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    padding: 1.1rem 1rem;
}
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #c8a96e;
    opacity: 0.7;
    letter-spacing: 0.1em;
    display: block;
    margin-bottom: 0.4rem;
}
.step-title {
    font-family: 'Manrope', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #ccc;
    display: block;
    margin-bottom: 0.3rem;
}
.step-desc {
    font-family: 'Manrope', sans-serif;
    font-size: 0.75rem;
    color: #444;
    line-height: 1.6;
}

/* ── Error ── */
.err {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #c07070;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    background: rgba(192,112,112,0.06);
    border: 1px solid rgba(192,112,112,0.15);
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-top: 0.5rem;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── TextRank Implementation ───────────────────────────────────────────────────

STOPWORDS = set([
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "not","with","this","that","was","are","be","by","as","from","has","have",
    "had","he","she","they","we","you","i","its","their","our","your","my",
    "will","can","do","did","so","if","up","out","about","into","than","then",
    "been","also","more","when","there","which","who","what","how","all","one",
    "no","were","just","would","could","should","these","those","after","before",
    "each","some","such","over","her","him","his","she","his","only","same"
])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def get_word_freq(sentences):
    freq = defaultdict(int)
    for sent in sentences:
        words = re.findall(r'\b[a-z]+\b', sent.lower())
        for w in words:
            if w not in STOPWORDS:
                freq[w] += 1
    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_freq
    return freq

def sentence_score(sentence, freq):
    words = re.findall(r'\b[a-z]+\b', sentence.lower())
    score = sum(freq.get(w, 0) for w in words if w not in STOPWORDS)
    return score / max(len(words), 1)

def cosine_similarity(s1, s2, freq):
    words1 = set(re.findall(r'\b[a-z]+\b', s1.lower())) - STOPWORDS
    words2 = set(re.findall(r'\b[a-z]+\b', s2.lower())) - STOPWORDS
    common = words1 & words2
    if not words1 or not words2:
        return 0.0
    return len(common) / (math.log(len(words1) + 1) + math.log(len(words2) + 1))

def textrank(sentences, num_sentences=3, damping=0.85, iterations=30):
    n = len(sentences)
    if n == 0:
        return []

    # Build similarity matrix
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = cosine_similarity(sentences[i], sentences[j], {})

    # Normalize rows
    for i in range(n):
        row_sum = sum(sim_matrix[i])
        if row_sum > 0:
            sim_matrix[i] = [v / row_sum for v in sim_matrix[i]]

    # PageRank iteration
    scores = [1.0 / n] * n
    for _ in range(iterations):
        new_scores = []
        for i in range(n):
            rank = sum(sim_matrix[j][i] * scores[j] for j in range(n))
            new_scores.append((1 - damping) / n + damping * rank)
        scores = new_scores

    # Rank and select top sentences (preserve original order)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:num_sentences])
    return [sentences[i] for i in top_indices]

def compression_ratio(original, summary):
    orig_words = len(original.split())
    summ_words = len(summary.split())
    if orig_words == 0:
        return 0
    return round((1 - summ_words / orig_words) * 100, 1)


# ─── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
    <div class="title-badge">NLP · Graph-Based Ranking</div>
    <h1>Text<em>Rank</em></h1>
    <div class="divider"></div>
    <p class="subtitle">Extractive Summarization · Pure Python · Zero Dependencies</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Input Text</div>', unsafe_allow_html=True)
input_text = st.text_area(
    label="input",
    label_visibility="collapsed",
    height=230,
    placeholder="Paste any article, essay, or paragraph here — minimum 2 sentences...",
)

col1, col2 = st.columns([3, 1])
with col1:
    num_sents = st.slider("Sentences in summary", min_value=1, max_value=10, value=3)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    word_count = len(input_text.split()) if input_text else 0
    st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#3a3a3a;text-align:right;padding-top:0.6rem;">{word_count} words</div>', unsafe_allow_html=True)

summarize_clicked = st.button("✦  Summarize Text")

if summarize_clicked:
    if not input_text or len(input_text.strip()) < 50:
        st.markdown('<div class="err">⚠ Please enter at least a few sentences of text.</div>', unsafe_allow_html=True)
    else:
        sentences = tokenize_sentences(input_text)
        if len(sentences) < 2:
            st.markdown('<div class="err">⚠ Not enough sentences detected. Try pasting a longer text.</div>', unsafe_allow_html=True)
        elif num_sents > len(sentences):
            st.markdown(f'<div class="err">⚠ Text only has {len(sentences)} sentences. Reduce the slider.</div>', unsafe_allow_html=True)
        else:
            summary_sents = textrank(sentences, num_sentences=num_sents)
            summary = " ".join(summary_sents)
            ratio = compression_ratio(input_text, summary)
            orig_words = len(input_text.split())
            summ_words = len(summary.split())

            # Build numbered sentences HTML
            sents_html = "".join(
                f'<span class="sent"><span class="sent-num">{i+1}</span>{s}</span>'
                for i, s in enumerate(summary_sents)
            )

            st.markdown(f"""
            <div class="output-wrapper">
                <div class="summary-header">
                    <span class="summary-tag">✦ Summary</span>
                    <span class="sentence-badge">{num_sents} of {len(sentences)} sentences</span>
                </div>
                <div class="summary-card">{sents_html}</div>
                <div class="stats-row">
                    <div class="stat-box">
                        <span class="stat-num">{orig_words}</span>
                        <span class="stat-label">Original Words</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-num">{summ_words}</span>
                        <span class="stat-label">Summary Words</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-num">{ratio}%</span>
                        <span class="stat-label">Compressed</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-num">{len(sentences)}</span>
                        <span class="stat-label">Sentences Found</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── How it works ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-it-works">
    <div class="section-label">How TextRank Works</div>
    <div class="steps-grid">
        <div class="step-card">
            <span class="step-num">Step 01</span>
            <span class="step-title">Sentence Tokenization</span>
            <span class="step-desc">Input is split into individual sentences and cleaned.</span>
        </div>
        <div class="step-card">
            <span class="step-num">Step 02</span>
            <span class="step-title">Similarity Matrix</span>
            <span class="step-desc">Cosine similarity is computed between every sentence pair using word overlap.</span>
        </div>
        <div class="step-card">
            <span class="step-num">Step 03</span>
            <span class="step-title">PageRank Scoring</span>
            <span class="step-desc">Sentences are scored iteratively — central sentences rank highest.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)