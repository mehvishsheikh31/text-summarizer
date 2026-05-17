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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f0f;
    color: #e8e4dc;
    font-family: 'Manrope', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 10% 20%, rgba(180,140,90,0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 80%, rgba(100,160,130,0.06) 0%, transparent 55%),
        #0f0f0f;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 780px; padding: 3rem 2rem 4rem; }

/* Title */
.title-block {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-top: 1rem;
}
.title-block h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #e8e4dc;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.title-block .accent { color: #c8a96e; }
.title-block p {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #5a5a5a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0;
}
.divider {
    width: 40px;
    height: 1px;
    background: #c8a96e;
    margin: 1rem auto;
    opacity: 0.6;
}

/* Input area */
.stTextArea textarea {
    background: #191919 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e8e4dc !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 1px rgba(200,169,110,0.2) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }
.stSlider .st-emotion-cache-1cxogv { color: #c8a96e !important; }
label[data-testid="stWidgetLabel"] {
    color: #9a9a9a !important;
    font-size: 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Button */
.stButton button {
    background: #c8a96e !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Output card */
.summary-card {
    background: #141414;
    border: 1px solid #242424;
    border-left: 3px solid #c8a96e;
    border-radius: 8px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    font-size: 0.97rem;
    line-height: 1.85;
    color: #ddd8ce;
    font-family: 'Manrope', sans-serif;
}
.summary-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #c8a96e;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.stat-box {
    flex: 1;
    background: #111;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #c8a96e;
    display: block;
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #555;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: block;
    margin-top: 0.3rem;
}

/* Error */
.err { color: #e07070; font-family: 'DM Mono', monospace; font-size: 0.82rem; }

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
    <h1>Text<span class="accent">Rank</span></h1>
    <div class="divider"></div>
    <p>Extractive Summarization · TextRank Algorithm</p>
</div>
""", unsafe_allow_html=True)

input_text = st.text_area(
    "Paste your text here",
    height=240,
    placeholder="Paste any article, paragraph, or document here...",
)

num_sents = st.slider("Number of sentences in summary", min_value=1, max_value=10, value=3)

summarize_clicked = st.button("✦  Summarize")

if summarize_clicked:
    if not input_text or len(input_text.strip()) < 50:
        st.markdown('<p class="err">⚠ Please enter at least a few sentences of text.</p>', unsafe_allow_html=True)
    else:
        sentences = tokenize_sentences(input_text)
        if len(sentences) < 2:
            st.markdown('<p class="err">⚠ Not enough sentences detected. Try pasting a longer text.</p>', unsafe_allow_html=True)
        elif num_sents > len(sentences):
            st.markdown(f'<p class="err">⚠ Text only has {len(sentences)} sentences. Reduce the slider.</p>', unsafe_allow_html=True)
        else:
            summary_sents = textrank(sentences, num_sentences=num_sents)
            summary = " ".join(summary_sents)
            ratio = compression_ratio(input_text, summary)
            orig_words = len(input_text.split())
            summ_words = len(summary.split())

            st.markdown(f"""
            <div class="summary-label">Summary</div>
            <div class="summary-card">{summary}</div>
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
            """, unsafe_allow_html=True)
