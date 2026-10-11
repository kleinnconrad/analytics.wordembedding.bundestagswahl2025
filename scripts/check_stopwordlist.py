"""Print the stop words removed in preprocessing: spaCy's German list plus config.CUSTOM_STOP_WORDS.

Run:  uv run python scripts/check_stopwordlist.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from textproc import get_nlp  # noqa: E402

stop_words = sorted(get_nlp().Defaults.stop_words)
print(f"{len(stop_words)} stop words:")
print(", ".join(stop_words))
