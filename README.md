# ✦ TextRank Text Summarizer

An extractive text summarization web app built with **TextRank algorithm** and **Streamlit** — no external NLP libraries required.

## 🧠 How It Works

TextRank is a graph-based ranking algorithm (inspired by Google's PageRank) applied to NLP:

1. **Sentence Tokenization** — Split input into individual sentences
2. **Similarity Matrix** — Compute cosine similarity between every sentence pair using word overlap
3. **Graph Construction** — Sentences = nodes, similarity scores = edge weights
4. **PageRank Iteration** — Iteratively score sentences by importance (damping factor = 0.85)
5. **Extraction** — Pick top-N ranked sentences, return in original order

> No transformers, no API calls, no heavy models — pure Python + math.

## 🚀 Features

- ✅ Pure Python TextRank (no NLTK, no spaCy dependency)
- ✅ Adjustable summary length via slider
- ✅ Live stats: original words, summary words, compression ratio
- ✅ Clean dark UI with DM Serif Display typography
- ✅ Works on articles, research abstracts, news, essays

## 📦 Installation

```bash
git clone https://github.com/mehvishsheikh31/text-summarizer
cd text-summarizer
pip install -r requirements.txt
streamlit run app.py
```

## 🗂️ Project Structure

```
text-summarizer/
├── app.py              # Streamlit app + TextRank logic
├── requirements.txt
└── README.md
```

## 📊 Example

**Input** (150 words article) → **Output** (3 key sentences, ~60% compression)

The algorithm scores each sentence based on how similar it is to all other sentences in the document. Sentences that share content with many others are considered more central/important.

## 🔬 Algorithm Details

| Parameter | Value |
|---|---|
| Similarity Metric | Cosine (word overlap) |
| Damping Factor | 0.85 |
| Iterations | 30 |
| Stopword Filtering | Yes (custom list) |
| Sentence Min Length | 20 chars |

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- Pure Python (math, re, collections)

## 👩‍💻 Author

**Mehvish Sheikh**  
B.Tech CSE (Data Science) · IIST Indore  
[LinkedIn](https://linkedin.com/in/mehvishsheikh31) · [GitHub](https://github.com/mehvishsheikh31)
