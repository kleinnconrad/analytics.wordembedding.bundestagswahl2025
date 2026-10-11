"""Tokenize the extracted election programs and train the Word2Vec models.

Every party is trained once per seed in config.SEEDS with the same parameters. The word
vectors are saved to models/<party>/seed_<n>.kv, the corpus statistics and parameters to
models/training_summary.json.

Run:  uv run python src/preprocess.py
"""

import json

import gensim
import spacy
from gensim.models import Word2Vec

from config import DATA_DIR, MANUAL_OVERRIDES, MODELS_DIR, PARTIES, SEEDS, SPACY_MODEL, WORD2VEC_PARAMS
from ensemble import model_path
from textproc import tokenize_corpus


def train_party(party):
    text = (DATA_DIR / f"{party}.txt").read_text(encoding="utf-8")
    sentences = tokenize_corpus(text)
    params = {**WORD2VEC_PARAMS, **MANUAL_OVERRIDES.get(party, {})}

    party_dir = MODELS_DIR / party
    party_dir.mkdir(parents=True, exist_ok=True)
    for old_model in party_dir.glob("seed_*.kv"):
        old_model.unlink()
    for seed in SEEDS:
        model = Word2Vec(sentences=sentences, seed=seed, **params)
        # gensim stores a timestamp per save; without it, identical models give identical files
        model.wv.lifecycle_events = None
        model.wv.save(str(model_path(party, seed)))

    return {
        "sentences": len(sentences),
        "tokens": sum(len(sentence) for sentence in sentences),
        "vocabulary": len(model.wv),
        "parameters": params,
    }


def run_preprocessing_and_training():
    summary = {
        "gensim": gensim.__version__,
        "spacy": spacy.__version__,
        "spacy_model": SPACY_MODEL,
        "seeds": SEEDS,
        "parties": {},
    }
    for party, name in PARTIES.items():
        if not (DATA_DIR / f"{party}.txt").exists():
            print(f"Skipping {name}: data/{party}.txt not found (run scripts/pdf_converter.py)")
            continue
        stats = train_party(party)
        summary["parties"][party] = stats
        overridden = sorted(MANUAL_OVERRIDES.get(party, {}))
        print(f"{name:8} {stats['sentences']:5} sentences, {stats['tokens']:6} tokens, "
              f"vocabulary {stats['vocabulary']:5}, {len(SEEDS)} seeds"
              + (f" (manual override: {', '.join(overridden)})" if overridden else ""))

    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Models and {summary_path.relative_to(MODELS_DIR.parent)} written")


if __name__ == "__main__":
    run_preprocessing_and_training()
