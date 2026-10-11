"""Show the nearest neighbours of a word or phrase in the model of every party.

Run:  uv run python src/query_tool.py "Innere Sicherheit" [--top 10]
"""

import argparse

from config import PARTIES, SEEDS, TOP_N
from ensemble import load_party
from textproc import tokenize_query


def run_query(phrase, top_n):
    tokens = tokenize_query(phrase)
    if not tokens:
        print(f"No words left in '{phrase}' after removing stop words.")
        return

    print(f"\nNearest neighbours of '{' '.join(tokens)}'")
    print(f"Similarity: mean cosine similarity over {len(SEEDS)} seed models. "
          f"Seeds: number of seed models with the word in their own top {top_n}.")
    print("=" * 72)

    for party, name in PARTIES.items():
        ensemble = load_party(party)
        if ensemble is None:
            print(f"{name}: models not found (run src/preprocess.py)")
            continue

        print(f"{name}:")
        neighbours = ensemble.neighbours(tokens, top_n)
        if neighbours is None:
            print("  Not in vocabulary")
        else:
            unknown = [t for t in tokens if t not in ensemble]
            if unknown:
                print(f"  Not in vocabulary, ignored: {', '.join(unknown)}")
            for word, similarity, seeds in neighbours:
                print(f"  {word:<30} similarity {similarity:.3f}   seeds {seeds}/{len(SEEDS)}")
        print("-" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the nearest neighbours of a phrase across the party models")
    parser.add_argument("phrase", help="word or phrase, e.g. 'Innere Sicherheit'")
    parser.add_argument("--top", type=int, default=TOP_N, help=f"number of neighbours (default {TOP_N})")
    args = parser.parse_args()
    run_query(args.phrase, args.top)
