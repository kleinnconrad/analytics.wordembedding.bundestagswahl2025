"""Extract the text of the election programs (corpora/*.pdf) into data/*.txt.

Steps per PDF:
  1. Remove duplicated characters (bold text that the PDF draws twice).
  2. Read two-column pages column by column (the Grüne program).
  3. Skip the cover and the CDU short version; drop table-of-contents pages, running headers
     and footers, and page numbers.
  4. Rejoin words that are hyphenated at a line, column or page break; write "CO₂" as "CO2".

Run:  uv run python scripts/pdf_converter.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CORPORA_DIR, DATA_DIR, PARTIES  # noqa: E402

# Pages (1-based) that are not part of the program text: the cover of every PDF, and the
# short version (Kurzfassung) at the start of cdu.pdf, which repeats the full version in brief
SKIPPED_PAGES = {
    "cdu": {1, *range(3, 12)},
    "gruene": {1},
    "spd": {1},
    "afd": {1},
}
# Lines at the top and bottom of a column that are checked for headers, footers and page numbers
EDGE_LINES = 2
# A column gutter is a vertical strip in the middle of the page that at most this many words cross
# (a heading that spans both columns) and that is at least MIN_GUTTER_WIDTH points wide
MAX_GUTTER_CROSSINGS = 2
MIN_GUTTER_WIDTH = 6
# A line that appears at a page edge on at least this many pages is a running header or footer
MIN_REPEATS = 5

PAGE_NUMBER = re.compile(r"^[\s\-–]*\d{1,3}[\s\-–]*$")
TOC_ENTRY = re.compile(r"(\.{3,}\s*|\s)\d{1,3}$")    # "Chapter ..... 12" or "Chapter 12"

# "CO₂" is written as "CO2". In the SPD program the subscript is extracted as a separate
# line "2" below the line with "CO -Preis" or "CO besonders".
CO2_BEFORE_HYPHEN = re.compile(r"\bCO -")
CO2_BEFORE_WORD = re.compile(r"\bCO (?=[a-zäöüß])")
CO2_SUBSCRIPT_LINE = re.compile(r"(CO2[^\n]*)\n2(?=\n)")

# A hyphen at a line break is removed when the line continues in lowercase ("Förde-\nrung"),
# kept before conjunctions ("Kinder-\nund Jugendhilfe") and kept in compounds ("EU-\nKommission")
HYPHEN_BEFORE_CONJUNCTION = re.compile(r"-\n(?=(?:und|oder|bzw|sowie|bis)\b)")
HYPHEN_IN_WORD = re.compile(r"(?<=\w)-\n(?=[a-zäöüß])")
HYPHEN_IN_COMPOUND = re.compile(r"(?<=\w)-\n(?=[A-ZÄÖÜ0-9])")


def edge_key(line):
    """Headers and footers differ only in their page or chapter number."""
    return re.sub(r"\d+", "#", line.strip())


def is_toc_page(lines):
    lines = [line.strip() for line in lines if line.strip()]
    entries = sum(1 for line in lines if TOC_ENTRY.search(line))
    # Entries that wrap onto a second line have no page number, hence the threshold below 0.5
    return len(lines) >= 5 and entries / len(lines) >= 0.4


def is_edge_noise(line, repeated):
    return edge_key(line) in repeated or PAGE_NUMBER.match(line) is not None


def find_gutter(page):
    """Return the x position between two text columns, or None for a single-column page."""
    words = page.extract_words()
    if len(words) < 40:
        return None
    crossings = [0] * (int(page.width) + 2)
    for word in words:
        for x in range(int(word["x0"]), int(word["x1"]) + 1):
            crossings[x] += 1

    best_x, best_width, run_start = None, 0, None
    for x in range(int(page.width * 0.3), int(page.width * 0.7) + 1):
        if crossings[x] > MAX_GUTTER_CROSSINGS:
            run_start = None
            continue
        run_start = x if run_start is None else run_start
        if x - run_start + 1 > best_width:
            best_x, best_width = (run_start + x) / 2, x - run_start + 1
    if best_width < MIN_GUTTER_WIDTH:
        return None

    left = sum(1 for word in words if word["x1"] <= best_x)
    right = sum(1 for word in words if word["x0"] >= best_x)
    return best_x if min(left, right) >= 0.2 * len(words) else None


def page_columns(page):
    """Return the lines of each text column of a page, from left to right."""
    gutter = find_gutter(page)
    if gutter is None:
        return [(page.extract_text() or "").splitlines()]
    halves = [(0, 0, gutter, page.height), (gutter, 0, page.width, page.height)]
    return [(page.crop(bbox).extract_text() or "").splitlines() for bbox in halves]


def extract_text(pdf_path, skipped_pages):
    pages, toc_pages = [], 0
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            if number in skipped_pages:
                continue
            page = page.dedupe_chars()
            # Checked before the column split, which can separate the entries from their page numbers
            if is_toc_page((page.extract_text() or "").splitlines()):
                toc_pages += 1
                continue
            pages.append(page_columns(page))

    edge_counts = Counter()
    for columns in pages:
        for lines in columns:
            edge_counts.update({edge_key(line) for line in lines[:EDGE_LINES] + lines[-EDGE_LINES:]})
    repeated = {key for key, count in edge_counts.items() if count >= MIN_REPEATS}

    bodies, two_column_pages = [], 0
    for columns in pages:
        two_column_pages += len(columns) == 2
        for lines in columns:
            start, end = 0, len(lines)
            while start < min(EDGE_LINES, end) and is_edge_noise(lines[start], repeated):
                start += 1
            while end > max(start, len(lines) - EDGE_LINES) and is_edge_noise(lines[end - 1], repeated):
                end -= 1
            bodies.append("\n".join(lines[start:end]))

    text = "\n".join(body for body in bodies if body)
    text = text.replace("₂", "2")
    text = CO2_BEFORE_HYPHEN.sub("CO2-", text)
    text = CO2_BEFORE_WORD.sub("CO2 ", text)
    text = CO2_SUBSCRIPT_LINE.sub(r"\1", text)
    text = HYPHEN_BEFORE_CONJUNCTION.sub("- ", text)
    text = HYPHEN_IN_WORD.sub("", text)
    text = HYPHEN_IN_COMPOUND.sub("-", text)
    return text, len(pages), toc_pages, two_column_pages


def convert_pdfs_to_txt():
    DATA_DIR.mkdir(exist_ok=True)

    for party in PARTIES:
        pdf_path = CORPORA_DIR / f"{party}.pdf"
        if not pdf_path.exists():
            print(f"Skipping {party}: {pdf_path} not found")
            continue

        skipped = SKIPPED_PAGES.get(party, set())
        text, used_pages, toc_pages, two_column_pages = extract_text(pdf_path, skipped)
        (DATA_DIR / f"{party}.txt").write_text(text + "\n", encoding="utf-8")
        print(f"{party:7} {used_pages:3} pages used ({two_column_pages} two-column), "
              f"{len(skipped)} skipped, {toc_pages} table of contents dropped, "
              f"{len(text.split()):6} words -> data/{party}.txt")


if __name__ == "__main__":
    convert_pdfs_to_txt()
