"""Load the trained models and query all seed models of a party together."""

from itertools import combinations

import numpy as np
from gensim.models import KeyedVectors

from config import MODELS_DIR, SEEDS, TOP_N


def model_path(party, seed):
    return MODELS_DIR / party / f"seed_{seed}.kv"


def load_party(party):
    """Return the ensemble of a party, or None if its models have not been trained."""
    paths = [model_path(party, seed) for seed in SEEDS]
    if not all(path.exists() for path in paths):
        return None
    return PartyEnsemble([KeyedVectors.load(str(path)) for path in paths])


class PartyEnsemble:
    """The models of one party: the same corpus and parameters, trained with different seeds.

    The seeds share the vocabulary but not the coordinate system, so vectors are never
    compared across seeds; only the similarities computed within each model are combined.
    """

    def __init__(self, models):
        self.vocab = models[0].index_to_key
        self.index = models[0].key_to_index
        # Unit vectors of every model, rows in the order of self.vocab
        self.vectors = [m.get_normed_vectors()[[m.key_to_index[w] for w in self.vocab]] for m in models]

    def __contains__(self, word):
        return word in self.index

    def __len__(self):
        return len(self.vocab)

    def _similarities(self, tokens):
        """Cosine similarity of every word to the phrase, one row per seed model.

        A phrase is represented by the mean of its token vectors. The tokens themselves are
        set to -inf so that they are never returned as their own neighbours.
        """
        rows = [self.index[t] for t in tokens]
        similarities = []
        for vectors in self.vectors:
            query = vectors[rows].mean(axis=0)
            similarities.append(vectors @ (query / np.linalg.norm(query)))
        similarities = np.array(similarities)
        similarities[:, rows] = -np.inf
        return similarities

    def neighbours(self, tokens, top_n=TOP_N):
        """Rank the vocabulary by the similarity to the phrase, averaged over the seed models.

        Tokens that are not in the vocabulary are ignored; returns None if none is known.
        Otherwise returns (word, mean similarity, seeds) tuples, where seeds is the number of
        seed models that rank the word among their own top_n neighbours.
        """
        known = [t for t in tokens if t in self]
        if not known:
            return None
        similarities = self._similarities(known)
        per_seed_top = [set(np.argpartition(-row, top_n)[:top_n]) for row in similarities]
        mean = similarities.mean(axis=0)
        ranking = np.argsort(-mean)[:top_n]
        return [(self.vocab[i], float(mean[i]), sum(i in top for top in per_seed_top)) for i in ranking]

    def seed_agreement(self, words, top_n=TOP_N):
        """Mean share of the top_n neighbours that two single seed models have in common."""
        overlaps = []
        for word in words:
            similarities = self._similarities([word])
            tops = [set(np.argpartition(-row, top_n)[:top_n]) for row in similarities]
            overlaps += [len(a & b) / top_n for a, b in combinations(tops, 2)]
        return float(np.mean(overlaps))
