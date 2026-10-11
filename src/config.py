"""Shared settings: paths, parties, stop words, training parameters and query terms."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPORA_DIR = PROJECT_ROOT / "corpora"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# File name -> display name. cdu.pdf is the joint program of CDU and CSU.
PARTIES = {
    "cdu": "CDU/CSU",
    "gruene": "Grüne",
    "spd": "SPD",
    "afd": "AfD",
}

SPACY_MODEL = "de_core_news_sm"

# Added to the spaCy stop word list: words that occur in every program regardless of topic
CUSTOM_STOP_WORDS = {
    "deutschland", "bund", "bundesregierung", "bundestag", "regierung",
    "wahlprogramm", "regierungsprogramm", "kapitel", "seite", "bzw", "sowie", "dabei",
    "cdu", "csu", "union", "grüne", "spd", "afd", "fdp",
    "müssen", "wollen", "sollen", "setzen", "werden",
    "unser", "unserer", "unseren", "jahr", "jahre", "ab",
}

# Identical for all parties, so that differences between the models come from the texts.
# The corpora are small (12k-23k tokens), so many epochs are needed for the vectors to converge.
# vector_size, window and epochs were chosen by comparing how many neighbours two models
# with different seeds have in common (see the README, section "Hyperparameters").
WORD2VEC_PARAMS = {
    "sg": 1,             # skip-gram
    "vector_size": 100,
    "window": 10,        # stop words are removed, so 10 positions span about one sentence
    "min_count": 3,
    "negative": 10,      # negative samples per context word
    "epochs": 100,
    "workers": 1,        # more than one thread makes training non-deterministic
}

# Per-party overrides for experiments, e.g. {"gruene": {"epochs": 50}}.
# Overrides make the models less comparable; leave empty for published results.
MANUAL_OVERRIDES = {party: {} for party in PARTIES}

# Every party is trained once per seed. Results are averaged over the seeds, because
# single models of corpora this small differ noticeably between seeds.
SEEDS = [1, 2, 3, 4, 5]

# Number of neighbours reported per term
TOP_N = 10

BUZZWORDS = [
    "Klimaschutz", "Innere Sicherheit", "Steuern", "Migration", "Wirtschaft",
    "Digitalisierung", "Bildung", "Rente", "Gesundheit", "Europa",
    "Infrastruktur", "Bürokratie", "Wohnen", "Familie", "Arbeitsplätze",
    "Soziale Gerechtigkeit", "Innovation", "Pflege", "Asyl", "Integration",
    "Verkehr", "Umweltschutz", "Mindestlohn", "Steuerentlastung", "Bundeswehr",
    "Demokratie", "Gleichstellung", "Ländlicher Raum", "Energiewende", "Datenschutz",
]
