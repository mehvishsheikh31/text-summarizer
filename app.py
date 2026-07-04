import streamlit as st
import re
import math
from collections import defaultdict

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ─── Modern NLP (Transformer) Support ──────────────────────────────────────────
# These give us real semantic understanding instead of raw word-overlap:
#   - sentence-transformers -> dense sentence embeddings for TextRank similarity
#   - transformers pipeline  -> BART/DistilBART abstractive summarization
#   - transformers pipeline  -> DistilBERT (SST-2) sentiment classifier
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_SUPPORT = True
except ImportError:
    EMBEDDINGS_SUPPORT = False

try:
    from transformers import pipeline
    TRANSFORMERS_SUPPORT = True
except ImportError:
    TRANSFORMERS_SUPPORT = False


@st.cache_resource(show_spinner=False)
def load_embedder():
    """Loads a small, fast sentence-embedding transformer (~80MB)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Loads a distilled BART model fine-tuned for abstractive summarization."""
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """Loads a DistilBERT model fine-tuned on SST-2 for sentiment classification."""
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

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

/* ── Sentiment badge ── */
.sentiment-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-top: 1rem;
}
.sentiment-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    border: 1px solid;
}
.sentiment-positive { color: #7ec87e; border-color: rgba(126,200,126,0.3); background: rgba(126,200,126,0.06); }
.sentiment-negative { color: #c87e7e; border-color: rgba(200,126,126,0.3); background: rgba(200,126,126,0.06); }
.sentiment-neutral  { color: #7e9ec8; border-color: rgba(126,158,200,0.3); background: rgba(126,158,200,0.06); }
.sentiment-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #333;
    letter-spacing: 0.1em;
}

/* ── Readability ── */
.readability-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.7rem;
    margin-top: 1rem;
}
.readability-box {
    background: #0a0a0a;
    border: 1px solid #161616;
    border-radius: 10px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.readability-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #c8a96e;
    display: block;
}
.readability-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.53rem;
    color: #303030;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    display: block;
    margin-top: 0.3rem;
}
.grade-badge {
    display: inline-block;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: #c8a96e;
    background: rgba(200,169,110,0.08);
    border: 1px solid rgba(200,169,110,0.2);
    border-radius: 10px;
    padding: 0.15rem 0.55rem;
    letter-spacing: 0.06em;
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
    grid-template-columns: repeat(4, 1fr);
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

/* ── Warning / info ── */
.warn-box {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #c8a96e;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    background: rgba(200,169,110,0.05);
    border: 1px solid rgba(200,169,110,0.15);
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    margin-top: 0.5rem;
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

/* ── Mobile responsive ── */
@media (max-width: 600px) {
    .title-block h1 { font-size: 2.8rem; }
    .stats-row { grid-template-columns: repeat(2, 1fr); }
    .readability-row { grid-template-columns: repeat(2, 1fr); }
    .steps-grid { grid-template-columns: 1fr; }
    .block-container { padding: 1rem 1rem 4rem; }
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Expanded Stopwords ─────────────────────────────────────────────────────────
STOPWORDS = set([
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "not","with","this","that","was","are","be","by","as","from","has","have",
    "had","he","she","they","we","you","i","its","their","our","your","my",
    "will","can","do","did","so","if","up","out","about","into","than","then",
    "been","also","more","when","there","which","who","what","how","all","one",
    "no","were","just","would","could","should","these","those","after","before",
    "each","some","such","over","her","him","his","only","same","very","may",
    "even","two","new","now","most","other","any","between","through","while",
    # Extended
    "said","say","says","get","got","make","made","made","like","time","need",
    "way","use","used","using","come","came","go","went","know","knew","think",
    "take","took","see","saw","look","looked","want","wanted","give","gave",
    "find","found","tell","told","ask","asked","seem","seemed","call","called",
    "keep","kept","let","put","set","show","showed","try","tried","feel","felt",
    "become","became","leave","left","mean","meant","start","started","turn",
    "follow","move","play","run","change","different","point","well","back",
    "first","last","long","great","little","own","right","big","high","place",
    "end","hand","large","small","next","early","young","important","public",
    "able","thing","things","people","man","woman","child","world","life","day",
    "year","years","work","part","place","case","week","company","system","program",
    "question","government","number","night","group","area","lot","side","problem",
    "per","cent","mr","ms","mrs","dr","still","must","many","much","often","yet",
    "again","though","since","without","within","always","never","ever","every",
    "nothing","something","anything","everything","however","therefore","thus",
    "although","unless","whether","against","around","because","during","including",
    "until","along","across","behind","beyond","plus","except","among"
])

# ─── Sentiment Word Lists ────────────────────────────────────────────────────────
POSITIVE_WORDS = set([
    "good","great","excellent","amazing","wonderful","fantastic","outstanding",
    "best","better","positive","success","successful","win","winning","achieve",
    "achievement","benefit","beneficial","effective","efficient","improve","improved",
    "improvement","increase","increased","growth","strong","strength","powerful",
    "innovative","innovation","creative","opportunity","opportunities","advance",
    "advanced","progress","progressive","helpful","useful","valuable","significant",
    "important","remarkable","impressive","exceptional","superior","perfect",
    "ideal","optimistic","hope","hopeful","promising","bright","brilliant","smart",
    "capable","confident","proud","happy","joy","joyful","love","beautiful",
    "safe","secure","reliable","trustworthy","sustainable","thriving","flourishing",
    "prosperous","abundant","healthy","energetic","enthusiastic","passionate",
    "dedicated","committed","innovative","pioneering","leading","breakthrough"
])

NEGATIVE_WORDS = set([
    "bad","terrible","awful","horrible","poor","worst","worse","negative","fail",
    "failure","failing","lose","losing","loss","problem","problems","issue","issues",
    "concern","concerns","risk","risks","danger","dangerous","harmful","damage",
    "damaged","decline","declined","decrease","decreased","weak","weakness","crisis",
    "difficult","difficulty","challenge","challenging","threat","threatening","fear",
    "fearful","worried","worry","anxious","anxiety","stress","stressful","trouble",
    "troubled","conflict","controversy","controversial","criticism","criticize",
    "criticizes","wrong","error","mistake","fault","blame","accused","corrupt",
    "corruption","violence","violent","attack","attacked","destroy","destroyed",
    "collapse","collapsing","fail","fatal","serious","severe","extreme","critical",
    "alarming","shocking","devastating","tragic","tragedy","disaster","catastrophe",
    "suffering","pain","struggle","struggling","lack","lacking","missing","absent",
    "inadequate","insufficient","ineffective","inefficient","unstable","uncertain",
    "unclear","confused","confusing","misleading","false","fake","fraud","scam"
])


# ─── NLP Functions ──────────────────────────────────────────────────────────────

# ── FIX 1: Smarter sentence tokenizer (handles abbreviations) ──
ABBREVS = r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|St|Ave|Corp|Inc|Ltd|Fig|No|Dept|Govt|Univ|Assoc|Est)\.'

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    """Improved tokenizer that handles abbreviations correctly."""
    text = clean_text(text)
    # Temporarily protect abbreviation periods
    text = re.sub(ABBREVS, lambda m: m.group().replace('.', '<DOT>'), text)
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Restore protected periods and filter short fragments
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def count_syllables(word):
    """Approximate syllable count for readability scoring."""
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    # Remove silent 'e' at end
    word = re.sub(r'e$', '', word)
    # Count vowel groups
    count = len(re.findall(r'[aeiou]+', word))
    return max(1, count)

def flesch_score(text):
    """
    Flesch Reading Ease score.
    90-100: Very easy, 70-80: Easy, 60-70: Standard,
    50-60: Fairly difficult, 30-50: Difficult, 0-30: Very confusing.
    """
    sentences = tokenize_sentences(text)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not sentences or not words:
        return 0, 0, "Unknown"

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(w) for w in words)

    if num_sentences == 0 or num_words == 0:
        return 0, 0, "Unknown"

    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = num_syllables / num_words

    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    score = max(0, min(100, round(score, 1)))

    if score >= 90:
        grade = "Very Easy"
    elif score >= 80:
        grade = "Easy"
    elif score >= 70:
        grade = "Fairly Easy"
    elif score >= 60:
        grade = "Standard"
    elif score >= 50:
        grade = "Fairly Hard"
    elif score >= 30:
        grade = "Difficult"
    else:
        grade = "Very Hard"

    return score, round(avg_sentence_length, 1), grade

def lexicon_sentiment(text):
    """Fallback: simple lexicon-based sentiment analysis (no model download)."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return "Neutral", 0.5, "lexicon"
    pos_ratio = pos_count / total
    if pos_ratio > 0.6:
        return "Positive", round(pos_ratio, 2), "lexicon"
    elif pos_ratio < 0.4:
        return "Negative", round(1 - pos_ratio, 2), "lexicon"
    else:
        return "Neutral", round(pos_ratio, 2), "lexicon"


def analyze_sentiment(text):
    """
    Transformer-based sentiment analysis using DistilBERT fine-tuned on
    SST-2. Falls back to lexicon scoring if transformers isn't installed.
    Long text is chunked (model has a 512-token limit) and chunk scores
    are averaged.
    """
    if not TRANSFORMERS_SUPPORT:
        return lexicon_sentiment(text)

    try:
        classifier = load_sentiment_model()
        # Chunk on sentence boundaries to respect the model's token limit
        chunks = tokenize_sentences(text) or [text]
        merged, buf = [], ""
        for s in chunks:
            if len(buf) + len(s) < 800:
                buf += " " + s
            else:
                merged.append(buf.strip())
                buf = s
        if buf.strip():
            merged.append(buf.strip())

        results = classifier(merged, truncation=True)
        pos_scores, neg_scores = [], []
        for r in results:
            if r["label"] == "POSITIVE":
                pos_scores.append(r["score"])
            else:
                neg_scores.append(r["score"])

        pos_total, neg_total = sum(pos_scores), sum(neg_scores)
        if pos_total > neg_total:
            confidence = pos_total / (len(pos_scores) or 1)
            label = "Positive" if confidence > 0.6 else "Neutral"
        else:
            confidence = neg_total / (len(neg_scores) or 1)
            label = "Negative" if confidence > 0.6 else "Neutral"

        return label, round(confidence, 2), "distilbert-sst2"
    except Exception:
        return lexicon_sentiment(text)

def get_word_freq(sentences):
    """TF-IDF inspired keyword scoring."""
    # Term frequency
    tf = defaultdict(int)
    doc_freq = defaultdict(int)
    total_docs = len(sentences)

    for sent in sentences:
        words = set(re.findall(r'\b[a-z]+\b', sent.lower()))
        sent_words = re.findall(r'\b[a-z]+\b', sent.lower())
        for w in sent_words:
            if w not in STOPWORDS and len(w) > 2:
                tf[w] += 1
        for w in words:
            if w not in STOPWORDS and len(w) > 2:
                doc_freq[w] += 1

    # TF-IDF score
    tfidf = {}
    for w, freq in tf.items():
        idf = math.log((total_docs + 1) / (doc_freq[w] + 1)) + 1
        tfidf[w] = freq * idf

    # Normalize
    max_score = max(tfidf.values()) if tfidf else 1
    for w in tfidf:
        tfidf[w] /= max_score
    return tfidf

# ── FIX 2: Proper cosine similarity ──
def cosine_similarity(s1, s2):
    """Standard cosine similarity using word vectors."""
    words1 = re.findall(r'\b[a-z]+\b', s1.lower())
    words2 = re.findall(r'\b[a-z]+\b', s2.lower())

    # Build frequency vectors
    vec1 = defaultdict(int)
    vec2 = defaultdict(int)
    for w in words1:
        if w not in STOPWORDS and len(w) > 2:
            vec1[w] += 1
    for w in words2:
        if w not in STOPWORDS and len(w) > 2:
            vec2[w] += 1

    if not vec1 or not vec2:
        return 0.0

    # Dot product
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in common)

    # Magnitudes
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def embedding_similarity_matrix(sentences):
    """
    Modern replacement for word-overlap similarity: encodes every sentence
    with a transformer (all-MiniLM-L6-v2) and scores pairs by cosine
    similarity of their dense embeddings. This captures meaning/paraphrase
    ("the feline sat" ~ "the cat rested") that word-overlap cannot.
    """
    embedder = load_embedder()
    vectors = embedder.encode(sentences, normalize_embeddings=True)
    n = len(sentences)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = float(np.dot(vectors[i], vectors[j]))
    return sim_matrix


# ── FIX 3: TextRank with early convergence ──
def textrank(sentences, num_sentences=3, damping=0.85, iterations=50, use_embeddings=False):
    n = len(sentences)
    if n == 0:
        return [], []

    if use_embeddings and EMBEDDINGS_SUPPORT:
        sim_matrix = embedding_similarity_matrix(sentences)
    else:
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i][j] = cosine_similarity(sentences[i], sentences[j])

    # Row normalize
    for i in range(n):
        row_sum = sum(sim_matrix[i])
        if row_sum > 0:
            sim_matrix[i] = [v / row_sum for v in sim_matrix[i]]

    scores = [1.0 / n] * n
    # ── FIX: Early convergence check ──
    for _ in range(iterations):
        new_scores = []
        for i in range(n):
            rank = sum(sim_matrix[j][i] * scores[j] for j in range(n))
            new_scores.append((1 - damping) / n + damping * rank)
        # Stop early if converged
        delta = max(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if delta < 1e-5:
            break

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:num_sentences])
    return [sentences[i] for i in top_indices], scores

def abstractive_summarize(text, max_length=130, min_length=30):
    """
    Modern abstractive summarization using a distilled BART model
    (sshleifer/distilbart-cnn-12-6). Unlike TextRank, this GENERATES new
    sentences rather than extracting existing ones — the hallmark of
    modern (transformer-based) NLP summarization.

    BART's encoder caps input around 1024 tokens, so long text is chunked,
    each chunk is summarized, and the chunk summaries are combined.
    """
    if not TRANSFORMERS_SUPPORT:
        return None

    summarizer = load_summarizer()
    words = text.split()
    chunk_size = 600  # ~ safely under the 1024-token BART limit
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    partial_summaries = []
    for chunk in chunks:
        out = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        partial_summaries.append(out[0]["summary_text"].strip())

    return " ".join(partial_summaries)


def compression_ratio(original, summary):
    orig_words = len(original.split())
    summ_words = len(summary.split())
    if orig_words == 0:
        return 0
    return round((1 - summ_words / orig_words) * 100, 1)

def highlight_keywords(text, keywords):
    result = text
    for kw in keywords:
        pattern = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
        result = pattern.sub(r"<mark class='kw'>\1</mark>", result)
    return result

def detect_language_hint(text):
    """Very basic check — warns if text looks non-English."""
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    total = len(text.split())
    if total == 0:
        return True
    english_ratio = len(words) / total
    return english_ratio > 0.5  # True = likely English


# ─── UI ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
    <div class="title-eyebrow">Modern NLP · Transformer-Powered</div>
    <h1>Text<em>Rank</em></h1>
    <div class="title-divider"></div>
    <p class="subtitle">Extractive (Semantic) + Abstractive (BART) Summarization · DistilBERT Sentiment</p>
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
            raw = uploaded_file.read()
            try:
                prefill = raw.decode("utf-8")
            except UnicodeDecodeError:
                prefill = raw.decode("latin-1")
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#c8a96e;'
                f'margin:-0.3rem 0 0.5rem;letter-spacing:0.08em;">✓ {fname} loaded — {len(prefill.split())} words</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.markdown(f'<div class="err">⚠ Could not read file: {e}</div>', unsafe_allow_html=True)

# ── Text Input ──
# FIX: Use session state so result persists when slider is moved
if "input_text" not in st.session_state:
    st.session_state["input_text"] = prefill
if prefill:
    st.session_state["input_text"] = prefill

input_text = st.text_area(
    label="input",
    label_visibility="collapsed",
    height=220,
    placeholder="Paste any article, essay, or paragraph here — minimum 2 sentences...",
    value=st.session_state["input_text"],
    key="input_box"
)

# Language hint
if input_text.strip() and not detect_language_hint(input_text):
    st.markdown(
        '<div class="warn-box">⚠ Text appears to be non-English. Results may be inaccurate — this tool is optimized for English.</div>',
        unsafe_allow_html=True
    )

# ── Mode Selector: Modern (Transformer) NLP ──
mode_options = ["Extractive · Semantic TextRank"]
if TRANSFORMERS_SUPPORT:
    mode_options.append("Abstractive · BART Transformer")

summary_mode = st.radio(
    "Summarization mode",
    mode_options,
    horizontal=True,
    help="Extractive uses sentence-embedding TextRank (picks real sentences). "
         "Abstractive uses a BART transformer to generate new sentences, like a human paraphrase."
)
use_embeddings = EMBEDDINGS_SUPPORT  # semantic similarity is used whenever the model is available

if not TRANSFORMERS_SUPPORT:
    st.markdown(
        '<div class="warn-box">ℹ Install `transformers` + `torch` to unlock abstractive '
        '(BART) summarization and transformer-based sentiment. See requirements.txt.</div>',
        unsafe_allow_html=True
    )

# ── Controls ──
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    num_sents = st.slider("Sentences in summary", min_value=1, max_value=15, value=3,
                          disabled=(summary_mode != mode_options[0]))
with col2:
    damping = st.slider("Damping factor", min_value=0.50, max_value=0.99, value=0.85, step=0.01,
                        help="PageRank damping — higher = more link-following weight",
                        disabled=(summary_mode != mode_options[0]))
with col3:
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.markdown(
        f'<div class="word-counter">{word_count}<br>words</div>',
        unsafe_allow_html=True
    )

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    show_keywords = st.checkbox("Keyword analysis", value=True)
with col_b:
    show_scores = st.checkbox("Sentence scores", value=True)
with col_c:
    show_extras = st.checkbox("Sentiment & Readability", value=True)

summarize_clicked = st.button("✦  Summarize Text")

# ── Processing ──
if summarize_clicked:
    if not input_text or len(input_text.strip()) < 50:
        st.markdown('<div class="err">⚠ Please enter at least a few sentences of text.</div>', unsafe_allow_html=True)
    else:
        sentences = tokenize_sentences(input_text)
        if len(sentences) < 2:
            st.markdown('<div class="err">⚠ Not enough sentences detected. Try pasting a longer text.</div>', unsafe_allow_html=True)
        else:
            is_abstractive = (summary_mode == "Abstractive · BART Transformer")

            if is_abstractive:
                with st.spinner("Generating summary with BART transformer..."):
                    summary = abstractive_summarize(input_text)
                summary_sents = tokenize_sentences(summary) if summary else [summary]
                all_scores = None
                actual_num = len(summary_sents)
            else:
                actual_num = min(num_sents, len(sentences))
                if num_sents > len(sentences):
                    st.markdown(
                        f'<div class="warn-box">ℹ Text has {len(sentences)} sentences — showing all {len(sentences)}.</div>',
                        unsafe_allow_html=True
                    )

                spinner_msg = "Ranking sentences (semantic embeddings)..." if use_embeddings else "Ranking sentences (word-overlap)..."
                with st.spinner(spinner_msg):
                    summary_sents, all_scores = textrank(
                        sentences, num_sentences=actual_num, damping=damping,
                        use_embeddings=use_embeddings
                    )

                summary = " ".join(summary_sents)

            ratio = compression_ratio(input_text, summary)
            orig_words = len(input_text.split())
            summ_words = len(summary.split())

            # Store in session state so it persists
            st.session_state["last_summary"] = summary
            st.session_state["last_ratio"] = ratio
            st.session_state["last_damping"] = damping
            st.session_state["last_num_sents"] = actual_num
            st.session_state["last_sentence_count"] = len(sentences)

            # Keywords via TF-IDF
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
                    <span class="sentence-badge">{
                        f"{actual_num} of {len(sentences)} sentences · d={damping}"
                        if not is_abstractive else
                        f"generated · BART transformer"
                    }</span>
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

            # ── Sentiment + Readability ──
            if show_extras:
                sentiment, confidence, sentiment_engine = analyze_sentiment(input_text)
                flesch, avg_sent_len, grade = flesch_score(input_text)
                sent_class = f"sentiment-{sentiment.lower()}"
                conf_pct = round(confidence * 100)
                engine_label = "DistilBERT (SST-2) transformer" if sentiment_engine == "distilbert-sst2" else "lexicon scoring (fallback)"

                st.markdown(f"""
                <div class="keywords-section">
                    <div class="section-label">Sentiment Analysis</div>
                    <div class="sentiment-row">
                        <span class="sentiment-badge {sent_class}">{sentiment}</span>
                        <span class="sentiment-label">{conf_pct}% confidence · {engine_label}</span>
                    </div>
                </div>
                <div class="keywords-section">
                    <div class="section-label">Readability</div>
                    <div class="readability-row">
                        <div class="readability-box">
                            <span class="readability-num">{flesch}</span>
                            <span class="readability-label">Flesch Score</span>
                            <span class="grade-badge">{grade}</span>
                        </div>
                        <div class="readability-box">
                            <span class="readability-num">{avg_sent_len}</span>
                            <span class="readability-label">Avg Sentence Length</span>
                        </div>
                        <div class="readability-box">
                            <span class="readability-num">{len(sentences)}</span>
                            <span class="readability-label">Total Sentences</span>
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
                    <div class="section-label">Top Keywords · TF-IDF</div>
                    <div class="kw-row">{chips}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Sentence Score Chart ──
            if show_scores and all_scores:
                max_score = max(all_scores) if all_scores else 1
                selected_set = set()
                ranked_idx = sorted(range(len(sentences)), key=lambda i: all_scores[i], reverse=True)
                for idx in ranked_idx[:actual_num]:
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

            # ── Download Buttons ──
            st.markdown("<br>", unsafe_allow_html=True)
            dl_col1, dl_col2, dl_col3, _ = st.columns([1, 1, 1, 1])
            with dl_col1:
                st.download_button(
                    label="↓  .txt",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
            with dl_col2:
                md_content = f"# Summary\n\n{summary}\n\n---\n*{actual_num} sentences · {ratio}% compressed · damping={damping}*"
                st.download_button(
                    label="↓  .md",
                    data=md_content,
                    file_name="summary.md",
                    mime="text/markdown"
                )
            with dl_col3:
                if show_extras:
                    report = (
                        f"# Text Analysis Report\n\n"
                        f"## Summary\n{summary}\n\n"
                        f"## Statistics\n"
                        f"- Original words: {orig_words}\n"
                        f"- Summary words: {summ_words}\n"
                        f"- Compression: {ratio}%\n"
                        f"- Sentences: {len(sentences)}\n\n"
                        f"## Sentiment\n- Label: {sentiment}\n- Confidence: {conf_pct}%\n\n"
                        f"## Readability\n- Flesch Score: {flesch} ({grade})\n"
                        f"- Avg sentence length: {avg_sent_len} words\n\n"
                        f"## Keywords\n{', '.join(top_keywords)}\n"
                    )
                    st.download_button(
                        label="↓  Report",
                        data=report,
                        file_name="analysis_report.md",
                        mime="text/markdown"
                    )

# ── Persist summary if slider changes ──
elif "last_summary" in st.session_state:
    s = st.session_state
    st.markdown(f"""
    <div class="output-wrapper">
        <div class="summary-header">
            <span class="summary-tag">Last Summary</span>
            <span class="sentence-badge">{s['last_num_sents']} of {s['last_sentence_count']} sentences</span>
        </div>
        <div class="summary-card">{s['last_summary']}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── How it works ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-it-works">
    <div class="section-label">How the Modern Pipeline Works</div>
    <div class="steps-grid">
        <div class="step-card">
            <span class="step-num">Step 01</span>
            <span class="step-title">Sentence Embeddings</span>
            <span class="step-desc">Each sentence is encoded by a MiniLM sentence-transformer into a dense vector that captures meaning, not just word overlap.</span>
        </div>
        <div class="step-card">
            <span class="step-num">Step 02</span>
            <span class="step-title">Semantic TextRank</span>
            <span class="step-desc">Cosine similarity between embeddings builds the sentence graph; PageRank scores which sentences are most central in meaning.</span>
        </div>
        <div class="step-card">
            <span class="step-num">Step 03</span>
            <span class="step-title">Abstractive Generation</span>
            <span class="step-desc">Optionally, a DistilBART transformer reads the full text and generates a brand-new, paraphrased summary — real language generation, not extraction.</span>
        </div>
        <div class="step-card">
            <span class="step-num">Step 04</span>
            <span class="step-title">Transformer Sentiment</span>
            <span class="step-desc">A DistilBERT model fine-tuned on SST-2 classifies tone with contextual understanding, replacing simple keyword counting.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)