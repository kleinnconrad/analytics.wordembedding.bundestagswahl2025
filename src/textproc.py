"""Text normalization and tokenization, shared by training and queries.

Training texts and query phrases go through the same steps, so that a query term
is reduced to exactly the token the models were trained on.
"""

import re
from functools import lru_cache

import spacy

from config import CUSTOM_STOP_WORDS, SPACY_MODEL

# Gender forms "Bürger*innen", "Bürger:innen", "Bürger_innen" and "BürgerInnen" -> "Bürgerinnen".
# spaCy does not split them, and as non-alphabetic tokens they would be dropped entirely.
GENDER_FORM = re.compile(r"(?<=[a-zäöüß])(?:[*:_]i|I)n(nen)?\b")
LINE_BREAK = re.compile(r"\s*\n\s*")
# A word: letters and digits, optionally joined by hyphens, with at least one letter
# ("Klimaschutz", "CO2", "EU-Kommission"). Pure numbers and punctuation are dropped.
WORD = re.compile(r"(?=[^\W_]*[^\W\d_])[^\W_]+(?:-[^\W_]+)*")


@lru_cache(maxsize=1)
def get_nlp():
    """Load the spaCy pipeline with the custom stop words and rule-based sentence splitting."""
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    for word in CUSTOM_STOP_WORDS:
        nlp.Defaults.stop_words.add(word)
        nlp.vocab[word].is_stop = True
    return nlp


def normalize(text):
    text = GENDER_FORM.sub(lambda match: "in" + (match.group(1) or ""), text)
    # PDF line breaks end lines, not sentences
    return LINE_BREAK.sub(" ", text)


def _lemmas(tokens):
    stop_words = get_nlp().Defaults.stop_words
    lemmas = []
    for token in tokens:
        lemma = token.lemma_.lower()
        if (len(lemma) > 1 and WORD.fullmatch(token.text) and WORD.fullmatch(lemma)
                and lemma not in stop_words and token.text.lower() not in stop_words):
            lemmas.append(lemma)
    return lemmas


def tokenize_corpus(text):
    """Split a corpus into sentences, each a list of lowercase lemmas without stop words."""
    doc = get_nlp()(normalize(text))
    sentences = (_lemmas(sentence) for sentence in doc.sents)
    return [sentence for sentence in sentences if sentence]


def tokenize_query(phrase):
    """Reduce a query phrase to the lemmas used in training.

    A capitalized adjective at the start of a term ("Ländlicher Raum") is not lemmatized by
    spaCy. In the programs the adjective is lowercase ("im ländlichen Raum"), so adjectives
    are lowercased and the phrase is processed again.
    """
    nlp = get_nlp()
    doc = nlp(normalize(phrase))
    text = "".join((t.text.lower() if t.pos_ == "ADJ" else t.text) + t.whitespace_ for t in doc)
    return _lemmas(nlp(text))
