import streamlit as st
import re
import math
from collections import defaultdict

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextRank Summarizer",
    page_icon="✦",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Manrope:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080808;
    color: #e8e4dc;
    font-family: 'Manrope', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 60% 40% at 20% 0%, rgba(200,169,110,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 100%, rgba(80,140,110,0.05) 0%, transparent 60%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 60px,
            rgba(255,255,255,0.012) 60px,
            rgba(255,255,255,0.012) 61px
        ),
        #080808;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 820px; padding: 2rem 2rem 6rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }

/* ── Title ── */
.title-block {
    text-align: center;
    margin-bottom: 3.5rem;
    padding-top: 2rem;
    position: relative;
}
.title-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #c8a96e;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    border: 1px solid rgba(200,169,110,0.25);
    border-radius: 20px;
    padding: 0.35rem 1rem;
    margin-bottom: 1.4rem;
    position: relative;
}
.title-eyebrow::before {
    content: '';
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #c8a96e;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}
.title-block h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    font-weight: 400;
    color: #e8e4dc;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.title-block h1 em {
    font-style: italic;
    color: #c8a96e;
}
.title-block .subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.67rem;
    color: #333;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.9rem;
}
.title-divider {
    width: 1px;
    height: 36px;
    background: linear-gradient(to bottom, transparent, rgba(200,169,110,0.5), transparent);
    margin: 1.2rem auto;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #383838;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1a1a1a, transparent);
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0d0d0d !important;
    border: 1px dashed #222 !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    transition: border-color 0.25s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(200,169,110,0.3) !important;
}
[data-testid="stFileUploader"] label {
    color: #444 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Input area ── */
.stTextArea textarea {
    background: #0d0d0d !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.8 !important;
    padding: 1.1rem 1.2rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: rgba(200,169,110,0.4) !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.06) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #2a2a2a !important; }

/* ── Controls row ── */
.controls-strip {
    background: #0d0d0d;
    border: 1px solid #181818;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 0.8rem 0;
    display: grid;
    gap: 0.6rem;
}
.word-counter {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #2d2d2d;
    text-align: right;
    padding-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0.3rem 0; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #2a2a2a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
}
label[data-testid="stWidgetLabel"] {
    color: #383838 !important;
    font-size: 0.65rem !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── Checkbox ── */
.stCheckbox label {
    color: #555 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
}

/* ── Button ── */
.stButton button {
    background: linear-gradient(135deg, #c8a96e 0%, #a8823a 100%) !important;
    color: #080808 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(200,169,110,0.18) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton button::after {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 60%) !important;
}
.stButton button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(200,169,110,0.26) !important;
}
.stButton button:active { transform: translateY(0) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    color: #555 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    width: auto !important;
    transition: border-color 0.2s, color 0.2s !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: rgba(200,169,110,0.4) !important;
    color: #c8a96e !important;
    opacity: 1 !important;
    transform: none !important;
}

/* ── Output ── */
.output-wrapper {
    margin-top: 2.2rem;
    animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

.summary-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.9rem;
}
.summary-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    color: #c8a96e;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.summary-tag::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #c8a96e;
}
.sentence-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #444;
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    padding: 0.2rem 0.75rem;
    letter-spacing: 0.08em;
}

.summary-card {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #c8a96e;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    font-size: 0.97rem;
    line-height: 2;
    color: #cdc8bc;
    font-family: 'Manrope', sans-serif;
    position: relative;
    overflow: hidden;
}
.summary-card::before {
    content: '❝';
    position: absolute;
    top: 1rem;
    right: 1.5rem;
    font-size: 4rem;
    color: rgba(200,169,110,0.04);
    font-family: serif;
    line-height: 1;
}
.sent { display: inline; }
.sent + .sent::before { content: ' '; }
.sent-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    color: #c8a96e;
    opacity: 0.5;
    vertical-align: super;
    margin-right: 3px;
}
mark.kw {
    background: rgba(200,169,110,0.13);
    color: #d4b278;
    border-radius: 3px;
    padding: 0 2px;
    font-weight: 600;
}

/* ── Stats ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
    margin-top: 1rem;
}
.stat-box {
    background: #0a0a0a;
    border: 1px solid #161616;
    border-radius: 10px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: border-color 0.25s, transform 0.2s;
    cursor: default;
}
.stat-box:hover {
    border-color: rgba(200,169,110,0.18);
    transform: translateY(-2px);
}
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #c8a96e;
    display: block;
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    color: #303030;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    display: block;
    margin-top: 0.4rem;
}

/* ── Keyword chips ── */
.keywords-section { margin-top: 1rem; }
.kw-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.5rem;
}
.kw-chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #c8a96e;
    background: rgba(200,169,110,0.07);
    border: 1px solid rgba(200,169,110,0.18);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    letter-spacing: 0.06em;
    transition: background 0.2s;
}
.kw-chip:hover { background: rgba(200,169,110,0.13); }
.kw-weight {
    color: rgba(200,169,110,0.4);
    font-size: 0.55rem;
    margin-left: 0.3rem;
}

/* ── Score chart bars ── */
.score-section { margin-top: 1rem; }
.score-bars { margin-top: 0.6rem; display: flex; flex-direction: column; gap: 0.45rem; }
.score-row {
    display: grid;
    grid-template-columns: 2.2rem 1fr 2.8rem;
    align-items: center;
    gap: 0.6rem;
}
.score-idx {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #333;
    text-align: right;
}
.score-track {
    height: 5px;
    background: #141414;
    border-radius: 3px;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #c8a96e, #d4b278);
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
.score-fill.selected {
    background: linear-gradient(90deg, #c8a96e, #f0c870);
    box-shadow: 0 0 6px rgba(200,169,110,0.4);
}
.score-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: #333;
    text-align: right;
}

/* ── How it works ── */
.how-it-works {
    margin-top: 3.5rem;
    border-top: 1px solid #121212;
    padding-top: 2.2rem;
}
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin-top: 1rem;
}
.step-card {
    background: #0a0a0a;
    border: 1px solid #161616;
    border-radius: 10px;
    padding: 1.2rem 1.1rem;
    transition: border-color 0.25s;
}
.step-card:hover { border-color: #2a2a2a; }
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: #c8a96e;
    opacity: 0.6;
    letter-spacing: 0.1em;
    display: block;
    margin-bottom: 0.5rem;
}
.step-title {
    font-family: 'Manrope', sans-serif;
    font-size: 0.83rem;
    font-weight: 600;
    color: #bbb;
    display: block;
    margin-bottom: 0.35rem;
}
.step-desc {
    font-family: 'Manrope', sans-serif;
    font-size: 0.74rem;
    color: #383838;
    line-height: 1.65;
}

/* ── Error ── */
.err {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #b07070;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    background: rgba(176,112,112,0.05);
    border: 1px solid rgba(176,112,112,0.12);
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    margin-top: 0.5rem;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
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
    "each","some","such","over","her","him","his","only","same","very","may",
    "even","two","new","now","most","other","any","between","through","while"
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
            if w not in STOPWORDS and len(w) > 2:
                freq[w] += 1
    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_freq
    return freq

def cosine_similarity(s1, s2):
    words1 = set(re.findall(r'\b[a-z]+\b', s1.lower())) - STOPWORDS
    words2 = set(re.findall(r'\b[a-z]+\b', s2.lower())) - STOPWORDS
    common = words1 & words2
    if not words1 or not words2:
        return 0.0
    return len(common) / (math.log(len(words1) + 1) + math.log(len(words2) + 1))

def textrank(sentences, num_sentences=3, damping=0.85, iterations=30):
    n = len(sentences)
    if n == 0:
        return [], []

    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = cosine_similarity(sentences[i], sentences[j])

    for i in range(n):
        row_sum = sum(sim_matrix[i])
        if row_sum > 0:
            sim_matrix[i] = [v / row_sum for v in sim_matrix[i]]

    scores = [1.0 / n] * n
    for _ in range(iterations):
        new_scores = []
        for i in range(n):
            rank = sum(sim_matrix[j][i] * scores[j] for j in range(n))
            new_scores.append((1 - damping) / n + damping * rank)
        scores = new_scores

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:num_sentences])
    return [sentences[i] for i in top_indices], scores

def compression_ratio(original, summary):
    orig_words = len(original.split())
    summ_words = len(summary.split())
    if orig_words == 0:
        return 0
    return round((1 - summ_words / orig_words) * 100, 1)

def highlight_keywords(text, keywords):
    """Wrap top keywords in <mark class='kw'> tags."""
    result = text
    for kw in keywords:
        pattern = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
        result = pattern.sub(r"<mark class='kw'>\1</mark>", result)
    return result


# ─── UI ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
    <div class="title-eyebrow">NLP · Graph-Based Ranking</div>
    <h1>Text<em>Rank</em></h1>
    <div class="title-divider"></div>
    <p class="subtitle">Extractive Summarization · Pure Python · Zero Dependencies</p>
</div>
""", unsafe_allow_html=True)

# ── File Upload ──
st.markdown('<div class="section-label">Upload or Paste</div>', unsafe_allow_html=True)

upload_types = ["txt", "pdf"] if PDF_SUPPORT else ["txt"]
upload_help = "Upload a .txt or .pdf file to auto-fill the text area"

uploaded_file = st.file_uploader(
    label="upload",
    label_visibility="collapsed",
    type=upload_types,
    help=upload_help
)

prefill = ""
if uploaded_file is not None:
    fname = uploaded_file.name
    try:
        if fname.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t.strip())
            prefill = " ".join(pages_text)
            prefill = re.sub(r'\s+', ' ', prefill).strip()
            page_count = len(reader.pages)
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#c8a96e;'
                f'margin:-0.3rem 0 0.5rem;letter-spacing:0.08em;">'
                f'✓ {fname} loaded — {page_count} pages · {len(prefill.split())} words</div>',
                unsafe_allow_html=True
            )
        else:
            prefill = uploaded_file.read().decode("utf-8")
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#c8a96e;'
                f'margin:-0.3rem 0 0.5rem;letter-spacing:0.08em;">✓ {fname} loaded — {len(prefill.split())} words</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.markdown(f'<div class="err">⚠ Could not read file: {e}</div>', unsafe_allow_html=True)

# ── Text Input ──
input_text = st.text_area(
    label="input",
    label_visibility="collapsed",
    height=220,
    placeholder="Paste any article, essay, or paragraph here — minimum 2 sentences...",
    value=prefill,
)

# ── Controls ──
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    num_sents = st.slider("Sentences in summary", min_value=1, max_value=10, value=3)
with col2:
    damping = st.slider("Damping factor", min_value=0.50, max_value=0.99, value=0.85, step=0.01,
                        help="PageRank damping — higher = more link-following weight")
with col3:
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.markdown(
        f'<div class="word-counter">{word_count}<br>words</div>',
        unsafe_allow_html=True
    )

col_a, col_b = st.columns([1, 1])
with col_a:
    show_keywords = st.checkbox("Show keyword analysis", value=True)
with col_b:
    show_scores = st.checkbox("Show sentence score chart", value=True)

summarize_clicked = st.button("✦  Summarize Text")

# ── Processing ──
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
            summary_sents, all_scores = textrank(sentences, num_sentences=num_sents, damping=damping)
            summary = " ".join(summary_sents)
            ratio = compression_ratio(input_text, summary)
            orig_words = len(input_text.split())
            summ_words = len(summary.split())

            # Top keywords
            freq = get_word_freq(sentences)
            top_keywords = sorted(freq, key=freq.get, reverse=True)[:8]
            top_5_kw = top_keywords[:5]

            # Build highlighted sentence HTML
            sents_html = ""
            for i, s in enumerate(summary_sents):
                highlighted = highlight_keywords(s, top_5_kw)
                sents_html += f'<span class="sent"><span class="sent-num">{i+1}</span>{highlighted}</span>'

            st.markdown(f"""
            <div class="output-wrapper">
                <div class="summary-header">
                    <span class="summary-tag">Summary</span>
                    <span class="sentence-badge">{num_sents} of {len(sentences)} sentences · d={damping}</span>
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
                        <span class="stat-label">Sentences</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Keyword Chips ──
            if show_keywords and top_keywords:
                chips = "".join(
                    f'<span class="kw-chip">{kw}<span class="kw-weight">{freq[kw]:.2f}</span></span>'
                    for kw in top_keywords
                )
                st.markdown(f"""
                <div class="keywords-section">
                    <div class="section-label">Top Keywords</div>
                    <div class="kw-row">{chips}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Sentence Score Chart ──
            if show_scores and all_scores:
                max_score = max(all_scores) if all_scores else 1
                selected_set = set()
                ranked_idx = sorted(range(len(sentences)), key=lambda i: all_scores[i], reverse=True)
                for idx in ranked_idx[:num_sents]:
                    selected_set.add(idx)

                bars_html = ""
                for i, score in enumerate(all_scores):
                    pct = (score / max_score) * 100
                    sel_class = "selected" if i in selected_set else ""
                    bars_html += f"""
                    <div class="score-row">
                        <span class="score-idx">S{i+1}</span>
                        <div class="score-track">
                            <div class="score-fill {sel_class}" style="width:{pct:.1f}%"></div>
                        </div>
                        <span class="score-val">{score:.3f}</span>
                    </div>
                    """

                st.markdown(f"""
                <div class="score-section">
                    <div class="section-label">Sentence PageRank Scores <span style="color:#222;font-size:0.55rem;margin-left:0.3rem">gold = selected</span></div>
                    <div class="score-bars">{bars_html}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Download Button ──
            st.markdown("<br>", unsafe_allow_html=True)
            dl_col1, dl_col2, _ = st.columns([1, 1, 2])
            with dl_col1:
                st.download_button(
                    label="↓  Download .txt",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
            with dl_col2:
                md_content = f"# Summary\n\n{summary}\n\n---\n*{num_sents} sentences · {ratio}% compressed · damping={damping}*"
                st.download_button(
                    label="↓  Download .md",
                    data=md_content,
                    file_name="summary.md",
                    mime="text/markdown"
                )

# ─── How it works ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-it-works">
    <div class="section-label">How TextRank Works</div>
    <div class="steps-grid">
        <div class="step-card">
            <span class="step-num">Step 01</span>
            <span class="step-title">Sentence Tokenization</span>
            <span class="step-desc">Input is split into individual sentences, cleaned, and filtered by minimum length.</span>
        </div>
        <div class="step-card">
            <span class="step-num">Step 02</span>
            <span class="step-title">Similarity Matrix</span>
            <span class="step-desc">Cosine similarity is computed between every sentence pair using word overlap and log-normalization.</span>
        </div>
        <div class="step-card">
            <span class="step-num">Step 03</span>
            <span class="step-title">PageRank Scoring</span>
            <span class="step-desc">Sentences are scored iteratively using the damping factor — central sentences rank highest.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)