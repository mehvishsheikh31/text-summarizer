# ✦ Text Summarizer — Modern NLP (Transformer-Powered)

A text summarization, sentiment, and readability web app built with **Streamlit**, upgraded from classical word-overlap TextRank to a **transformer-based NLP pipeline**: sentence embeddings, abstractive generation, and a fine-tuned sentiment classifier.

## 🧠 How It Works

The app now supports two summarization modes:

### 1. Extractive — Semantic TextRank
Same graph-ranking idea as before, but the similarity between sentences is no longer word overlap — it's **meaning**:

1. **Sentence Tokenization** — Split input into individual sentences
2. **Sentence Embeddings** — Each sentence is encoded into a dense vector using `all-MiniLM-L6-v2` (a distilled sentence-transformer)
3. **Semantic Similarity Matrix** — Cosine similarity between *embeddings*, not raw words, so paraphrases ("the feline sat" ≈ "the cat rested") are correctly recognized as similar
4. **Graph Construction + PageRank** — Sentences = nodes, semantic similarity = edge weights, scored iteratively (damping factor adjustable)
5. **Extraction** — Pick top-N ranked sentences, return in original order

### 2. Abstractive — BART Transformer
Instead of picking existing sentences, a **DistilBART** model (`sshleifer/distilbart-cnn-12-6`) reads the text and **generates a brand-new summary in its own words** — genuine natural language generation, the hallmark of modern NLP. Long input is automatically chunked to respect the model's token limit.

### Sentiment Analysis
Upgraded from lexicon/keyword counting to `distilbert-base-uncased-finetuned-sst-2-english`, a DistilBERT model fine-tuned on the SST-2 sentiment benchmark — it understands context and negation ("not bad" ≠ "bad"), not just keyword presence. Falls back to the original lexicon method automatically if `transformers` isn't installed.

## 🚀 Features

- ✅ **Extractive** semantic summarization (sentence-transformer embeddings + TextRank)
- ✅ **Abstractive** summarization (DistilBART transformer — generates new text)
- ✅ **Transformer sentiment** classification (DistilBERT / SST-2) with lexicon fallback
- ✅ PDF/TXT upload, adjustable summary length, live compression stats
- ✅ Readability scoring (Flesch Reading Ease)
- ✅ TF-IDF keyword extraction + highlighting
- ✅ Downloadable `.txt` / `.md` / full analysis report
- ✅ Graceful degradation — works even without GPU or with only `streamlit` installed (falls back to the original pure-Python methods)

## 📦 Installation

```bash
git clone https://github.com/mehvishsheikh31/text-summarizer
cd text-summarizer
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ First run will download the transformer models (~300–500MB total: MiniLM embedder + DistilBART + DistilBERT). They're cached locally afterward via `st.cache_resource`, so subsequent runs are fast. A GPU is optional — CPU works fine for this scale.

## 🗂️ Project Structure

```
text-summarizer/
├── app.py              # Streamlit app + semantic TextRank + BART + DistilBERT sentiment
├── requirements.txt     # Now includes torch, transformers, sentence-transformers
└── README.md
```

## 🔬 Pipeline Details

| Component | Classical (before) | Modern (now) |
|---|---|---|
| Sentence similarity | Word-overlap cosine similarity | `all-MiniLM-L6-v2` sentence embeddings |
| Summarization | Extractive only | Extractive **+** Abstractive (DistilBART) |
| Sentiment | Hand-built positive/negative word lists | Fine-tuned DistilBERT (SST-2) classifier |
| Keyword extraction | TF-IDF (unchanged) | TF-IDF (unchanged) |
| Readability | Flesch Reading Ease (unchanged) | Flesch Reading Ease (unchanged) |

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- 🤗 `transformers` (BART, DistilBERT)
- `sentence-transformers` (MiniLM embeddings)
- PyTorch
- Pure Python fallback (math, re, collections) when models aren't installed

## 👩‍💻 Author

**Mehvish Sheikh**
B.Tech CSE (Data Science) · IIST Indore
[LinkedIn](https://linkedin.com/in/mehvishsheikh31) · [GitHub](https://github.com/mehvishsheikh31)