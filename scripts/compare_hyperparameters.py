"""Compare Word2Vec settings by how stable and how spread out the resulting vectors are.

For every combination of epochs, vector_size and window, each party is trained with two seeds.
Reported per combination, as the mean over the four parties:
  * Seed overlap: share of the top 10 neighbours that the two seed models have in common,
    for the buzzwords and for 60 of the 300 most frequent words. Higher is more stable.
  * Random-pair similarity: mean cosine similarity of 400 random word pairs. Values close to 1
    mean that all vectors point in the same direction (collapsed, undertrained vectors).
The remaining parameters are taken from config.WORD2VEC_PARAMS.

Run:  uv run python scripts/compare_hyperparameters.py   (takes about 15 minutes)
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import BUZZWORDS, DATA_DIR, PARTIES, WORD2VEC_PARAMS  # noqa: E402
from textproc import tokenize_corpus, tokenize_query  # noqa: E402

EPOCHS = [50, 100, 200]
VECTOR_SIZES = [50, 100]
WINDOWS = [5, 10]
TOP_N = 10


def top_words(model, word):
    return {w for w, _ in model.wv.most_similar(word, topn=TOP_N)}


def seed_overlap(a, b, words):
    return np.mean([len(top_words(a, w) & top_words(b, w)) / TOP_N for w in words])


def random_pair_similarity(model, rng, n=400):
    vectors = model.wv.get_normed_vectors()
    sample = vectors[rng.choice(len(vectors), min(n, len(vectors)), replace=False)]
    similarities = sample @ sample.T
    return similarities[~np.eye(len(sample), dtype=bool)].mean()


def main():
    corpora = {p: tokenize_corpus((DATA_DIR / f"{p}.txt").read_text(encoding="utf-8")) for p in PARTIES}
    buzzword_tokens = {t for phrase in BUZZWORDS for t in tokenize_query(phrase)}

    print("| Epochs | Vector size | Window | Seed overlap, buzzwords | Seed overlap, frequent words | Random-pair similarity |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for epochs, vector_size, window in itertools.product(EPOCHS, VECTOR_SIZES, WINDOWS):
        params = {**WORD2VEC_PARAMS, "epochs": epochs, "vector_size": vector_size, "window": window}
        rng = np.random.default_rng(0)
        rows = []
        for sentences in corpora.values():
            a = Word2Vec(sentences=sentences, seed=1, **params)
            b = Word2Vec(sentences=sentences, seed=2, **params)
            buzzwords = [w for w in buzzword_tokens if w in a.wv]
            frequent = list(rng.choice(a.wv.index_to_key[:300], 60, replace=False))
            rows.append((seed_overlap(a, b, buzzwords), seed_overlap(a, b, frequent),
                         random_pair_similarity(a, rng)))
        buzz, freq, spread = np.mean(rows, axis=0)
        print(f"| {epochs} | {vector_size} | {window} | {buzz:.0%} | {freq:.0%} | {spread:.2f} |", flush=True)


if __name__ == "__main__":
    main()
