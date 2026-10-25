# Political Word Embeddings: Election Programs Bundestagswahl 2025

This repository implements a comparative NLP pipeline for the election programs of CDU/CSU, Grüne, SPD and AfD for the 2025 Bundestag election. A separate Word2Vec model is trained on each program. For a political term, each model returns the words that occur in the most similar contexts in that program, which shows in which context each party uses the term.

[![Dashboard](https://img.shields.io/badge/Open-Interactive_Dashboard-2ea44f?style=for-the-badge&logo=html5)](https://kleinnconrad.github.io/analytics.wordembedding.bundestagswahl2025/)

The dashboard compares the nearest neighbours of 30 preset terms and of every lemma that occurs at least three times in all four programs, and explains the model step by step. The motivation is described in [docs/intro_doc.md](docs/intro_doc.md).

---

## Table of Contents
1. [Theoretical Framework](#theoretical-framework)
2. [Data Pipeline](#data-pipeline)
    * [Text Extraction](#text-extraction)
    * [Linguistic Preprocessing](#linguistic-preprocessing)
3. [Gensim Model Architecture](#gensim-model-architecture)
    * [The Objective Function](#the-objective-function)
    * [CBOW vs. Skip-gram](#cbow-vs-skip-gram)
    * [Negative Sampling](#negative-sampling)
    * [Hyperparameters](#hyperparameters)
    * [Seed Ensembles](#seed-ensembles)
4. [Similarity Metric](#similarity-metric)
5. [Results](#results)
6. [Limitations](#limitations)
7. [Lessons Learned](#lessons-learned)
8. [Repository Structure](#repository-structure)
9. [Setup and Usage](#setup-and-usage)
10. [Data Sources](#data-sources)

---

## Theoretical Framework
The project uses **Word2Vec**, a shallow neural network with one hidden layer, to map words to vectors in a continuous space $\mathbb{R}^d$. Words that occur in similar contexts receive similar vectors (distributional hypothesis). Because each model is trained on one party's program only, the vector space reflects how that party uses its vocabulary.

## Data Pipeline
The pipeline has four steps, each a script:

| Step | Script | Output |
| :--- | :--- | :--- |
| 1. Text extraction | [`scripts/pdf_converter.py`](scripts/pdf_converter.py) | `data/<party>.txt` |
| 2. Preprocessing and training | [`src/preprocess.py`](src/preprocess.py) | `models/<party>/seed_<n>.kv`, `models/training_summary.json` |
| 3. Batch analysis | [`src/batch_analysis.py`](src/batch_analysis.py) | `results/results.json` (dashboard data), `results/batch_results.md` |
| 4. Interactive queries | [`src/query_tool.py`](src/query_tool.py) | console output |

Shared settings (parties, stop words, hyperparameters, seeds, buzzwords) are in [`src/config.py`](src/config.py). The text processing used for both training and queries is in [`src/textproc.py`](src/textproc.py).

### Text Extraction
PDF parsing uses `pdfplumber`. The election programs differ in layout, so the extraction handles:
* **Duplicated characters:** The CDU/CSU program draws bold headings twice, which `pdfplumber` extracts as "PPoolliittiikk". `dedupe_chars()` removes the duplicates.
* **Two-column pages:** 142 of 160 pages of the Grüne program have two columns. A page without column detection is read line by line across both columns, which interleaves unrelated sentences. The converter finds the gap between the columns and reads each column separately.
* **Headers, footers and page numbers:** Lines that repeat at the top or bottom of at least five pages (e.g. "Regierungsprogramm 2025 Kapitel 3") and lines that contain only a page number are removed.
* **Skipped pages:** The cover of every program, table-of-contents pages, and the short version (Kurzfassung, pages 3 to 11) of the CDU/CSU program, which summarizes the full version and would count its content twice.
* **Hyphenation:** A hyphen at a line break is removed if the next line continues in lowercase ("Förde-rung"), and kept in compounds ("EU-Kommission") and before conjunctions ("Kinder- und Jugendhilfe").
* **CO₂:** The subscript is written as "CO2", so that all programs use the same token.

### Linguistic Preprocessing
* **Sentence segmentation:** The text is split into sentences. Each sentence is one training sequence, so the context window never extends across sentence boundaries.
* **Lemmatization:** spaCy's `de_core_news_sm` (a CNN-based pipeline) maps inflected forms to their lemma, e.g. "Steuern" to "steuer" and "Arbeitsplätze" to "arbeitsplatz". Lemmatization does not merge derived words or compounds: "steuerlich" and "Steuerlast" remain separate tokens.
* **Gender forms:** "Bürger\*innen", "Bürger:innen", "Bürger_innen" and "BürgerInnen" are normalized to "Bürgerinnen". Without this step, spaCy treats them as non-alphabetic tokens and they would be dropped, which affects the Grüne program (225 occurrences) far more than the others.
* **Token filter:** Tokens consisting of letters, digits and inner hyphens with at least one letter are kept ("klimaschutz", "co2-preis", "eu-kommission"). Punctuation and pure numbers are removed.
* **Stop words:** spaCy's German stop word list is extended by words that occur in every program regardless of topic (e.g. *Deutschland*, *Regierungsprogramm*, party names). Without them, the vector space is dominated by these frequent tokens.

---

## Gensim Model Architecture

### The Objective Function
For a sequence of training words $w_1, w_2, \dots, w_T$, the skip-gram model maximizes the average log probability of the context words:

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log p(w_{t+j} | w_t)$$

where $c$ is the context window size. The input layer is a one-hot vector, and the hidden layer has no activation function. The hidden layer therefore selects one row of the input weight matrix $W$, and **these rows are the word vectors**. The output weight matrix $W'$ is discarded after training.

### CBOW vs. Skip-gram
* **CBOW (Continuous Bag of Words):** The model predicts the center word $w_t$ from the average of the surrounding context vectors. It trains faster and works well for frequent words.
* **Skip-gram:** The model uses the center word to predict each surrounding context word. It produces better vectors for rare words and for small amounts of training data (Mikolov et al., 2013).
* **Project decision:** All models use **skip-gram** (`sg=1`), because each corpus contains only 12,000 to 23,000 tokens.

### Negative Sampling
Computing $p(w_{t+j} | w_t)$ with a softmax over the whole vocabulary is expensive. Gensim uses **negative sampling** instead: for each pair of input word $w_I$ and context word $w_O$, the model maximizes

$$\log \sigma({v'_{w_O}}^\top v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-{v'_{w_i}}^\top v_{w_I}) \right]$$

where $\sigma$ is the sigmoid function and the $k$ negative samples $w_i$ are drawn from the unigram distribution raised to the power of 3/4. Only the vectors of the context word and of the $k$ negative samples are updated per step. The dashboard shows this step for a sentence from the CDU/CSU program.

### Hyperparameters
All parties use the same parameters, so that differences between the models come from the texts and not from the settings. The four corpora are of the same order of magnitude (11,522 to 22,686 tokens), so a common setting fits all of them.

| Parameter | Value | Reason |
| :--- | :--- | :--- |
| `sg` | 1 | Skip-gram, see above |
| `vector_size` | 100 | More stable neighbours across seeds than 50 dimensions (see comparison below) |
| `window` | 10 | Stop words are removed, so 10 positions cover about one sentence |
| `min_count` | 3 | Words with fewer occurrences have too few contexts for a usable vector |
| `negative` | 10 | Negative samples per context word; 5 to 20 is the recommended range for small corpora |
| `epochs` | 100 | Small corpora need many passes; with 7 to 12 epochs, the vectors did not converge (see [Lessons Learned](#lessons-learned)) |
| `workers` | 1 | Multiple threads make the training order, and therefore the result, non-deterministic |

`vector_size`, `window` and `epochs` were chosen with [`scripts/compare_hyperparameters.py`](scripts/compare_hyperparameters.py). It trains each party with two seeds and measures how many of the top 10 neighbours the two models have in common (seed overlap), and the mean cosine similarity of random word pairs. A value close to 1 for the latter means that all vectors point in the same direction.

<details>
<summary>Comparison of the settings (mean over the four parties)</summary>

| Epochs | Vector size | Window | Seed overlap, buzzwords | Seed overlap, frequent words | Random-pair similarity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 50 | 5 | 66% | 64% | 0.40 |
| 50 | 50 | 10 | 72% | 70% | 0.34 |
| 50 | 100 | 5 | 77% | 73% | 0.42 |
| 50 | 100 | 10 | 80% | 80% | 0.34 |
| 100 | 50 | 5 | 68% | 64% | 0.27 |
| 100 | 50 | 10 | 70% | 67% | 0.26 |
| 100 | 100 | 5 | 79% | 77% | 0.26 |
| **100** | **100** | **10** | **81%** | **79%** | **0.25** |
| 200 | 50 | 5 | 62% | 58% | 0.26 |
| 200 | 50 | 10 | 65% | 60% | 0.25 |
| 200 | 100 | 5 | 77% | 75% | 0.25 |
| 200 | 100 | 10 | 79% | 78% | 0.24 |

100 dimensions are more stable than 50 in every combination, and a window of 10 more stable than 5. With 50 epochs, the vectors are less spread out (random-pair similarity 0.34); 200 epochs bring no further gain.

</details>

`MANUAL_OVERRIDES` in [`src/config.py`](src/config.py) allows per-party settings for experiments. Overrides make the models less comparable and are empty for the published results.

### Seed Ensembles
Word2Vec initializes the vectors randomly and draws random negative samples. On corpora of this size, two models trained with different seeds share only 73 % to 85 % of their top 10 neighbours. Each party is therefore trained with five seeds:
* The neighbours of a term are ranked by their cosine similarity **averaged over the five models**. The vectors of different seeds are never compared directly, because each model has its own coordinate system.
* For each neighbour, the number of models that rank it in their own top 10 is reported. Neighbours that fewer than three of five models return are marked as unstable.

---

## Similarity Metric
Similarity is measured with the **cosine similarity**, the cosine of the angle between two vectors. It depends only on the direction of the vectors, not on their length (which correlates with word frequency):

$$\text{Cosine Similarity} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}, \qquad \text{Cosine Distance} = 1 - \text{Cosine Similarity}$$

A phrase such as "Innere Sicherheit" is represented by the mean of its token vectors.

The values are comparable within one party's model. Across parties, the neighbour lists are compared, not the exact values, because each model has its own vector space. For this reason, the results are not mapped to fixed categories such as "strong association": thresholds of this kind would need a calibration that a corpus of this size does not allow.

---

## Results
The full results are in the [dashboard](https://kleinnconrad.github.io/analytics.wordembedding.bundestagswahl2025/) and in [`results/batch_results.md`](results/batch_results.md). The five nearest neighbours of four of the 30 terms:

| Term | CDU/CSU | Grüne | SPD | AfD |
| :--- | :--- | :--- | :--- | :--- |
| Migration | illegal, stopp, stoppen, geopolitisch, maß | menschenrechtsbasiert, migrationsabkommen, wertvoll, schutzsuchende, abschließen | kontrolle, paradigmenwechsel, begrenzen, humanitär, verfahren | compact, charta, begrüßen, vereint, beschlossen |
| Familie | traum, zusammenhalten, familienleistung, leistungsträger, eigenheim | kinderfreibetrag, aufteilung, sorgearbeit, vereinbarkeit, kindergeld | kinderzuschlag, alleinerziehend, kindergeld, belasten, partnerschaftlichkeit | ehe, belastung, teils, familiensplitting, außen |
| Rente | langfristig, einkommen, altersvorsorge, verlässlichkeit, generationengerechtigkeit | einzahlen, beamtinn, anspruchsvoll, mindestlohn, altersvorsorge | sinken, absolut, kaufkraft, stabilisieren, lebenshaltungskosten | rentenbeitrag, unterschied, kapitalmarkt, rentner, beziehen |
| Steuern | abgabe, lohn, belastung, wachstum, energie | zahlen, migrationsabkommen, gehalt, absenkung, lebenshaltungskosten | schenkungsteuer, grundsteuer, vermögensteuer, grund, kaufen | kostengünstig, erheben, erhebung, verbinden, abgabe |

The neighbours show the context of a term in a program, e.g. the AfD neighbours of "Migration" refer to the UN Global Compact for Migration. Reading the neighbours as a party's position requires checking the source sentences.

---

## Limitations
1. **Small corpora:** Each program contains 12,000 to 23,000 tokens after preprocessing; general-purpose embeddings are trained on billions. The neighbours indicate how a term is used, they do not prove intent.
2. **Variance:** Small changes of the parameters or the seed change part of the neighbours. The seed ensembles reduce this effect and make it visible, but do not remove it.
3. **No pre-training:** The models are trained from scratch to capture only the language of each program. The lemmatizer, however, is a general-purpose model and makes errors, e.g. "Beamt\*innen" becomes "beamtinn".
4. **Similarity is not synonymy:** Words with opposite meanings (e.g. "erhöhen" and "senken") occur in the same contexts and therefore have similar vectors.
5. **Vocabulary:** Terms with fewer than three occurrences in a program have no vector. "Steuerentlastung" is in none of the four vocabularies, "Asyl" only in the Grüne program.

---

## Lessons Learned
The first version of this pipeline produced neighbours that did not depend on the query term. The causes and fixes:

| Problem | Effect | Fix |
| :--- | :--- | :--- |
| Every PDF line was one training sequence | 4 to 7 tokens per sequence after filtering; sentences were cut at every line break | Sentence segmentation of the full text |
| 7 to 12 epochs on small corpora | Undertrained vectors: the mean cosine similarity of random word pairs was 0.81 to 0.98, so most queries returned cosine distances of 0.001 to 0.03 for unrelated words | 100 epochs; random-pair similarity 0.25 |
| Different hyperparameters per corpus size | Models not comparable; the rule gave large corpora fewer epochs | One setting for all parties |
| Bold text drawn twice in the CDU/CSU PDF | 3,296 garbled words (about 12 % of the corpus), e.g. "ddeennkkeenn" | `dedupe_chars()` |
| Two-column layout of the Grüne PDF | Lines of both columns interleaved on 142 pages; words hyphenated at a line end were not rejoined ("rung", "gen") | Column detection |
| Fixed crop of 10 % at the top and bottom of each page | Body text near the page edges removed, e.g. in the CDU/CSU, SPD and AfD programs | Removal of repeated headers and footers instead of cropping |
| Token filter `is_alpha` | Gender forms, "CO2" and hyphenated compounds removed | Normalization and a word pattern instead of `is_alpha` |
| Training with four threads, no seed control | Different models on every run | `workers=1`, fixed seeds, five seeds per party |

---

## Repository Structure

| Path | Content |
| :--- | :--- |
| [`corpora/`](corpora/) | Election programs as published (PDF) |
| [`data/`](data/) | Extracted text, one file per party |
| [`models/`](models/) | Trained word vectors (`<party>/seed_<n>.kv`) and `training_summary.json` |
| [`results/`](results/) | `results.json` (dashboard data) and `batch_results.md` |
| [`scripts/`](scripts/) | PDF extraction, hyperparameter comparison, stop word list |
| [`src/`](src/) | Configuration, text processing, training, analysis, query tool |
| [`docs/intro_doc.md`](docs/intro_doc.md) | Motivation and approach |
| [`index.html`](index.html) | Dashboard (GitHub Pages) |

---

## Setup and Usage

### Setup
The project uses [uv](https://docs.astral.sh/uv/). `pyproject.toml` declares the dependencies and the supported Python versions (3.12 and 3.13); `uv.lock` pins the complete dependency tree. The spaCy model `de_core_news_sm` is a regular dependency, so no separate download is needed.

```bash
uv sync
```

After changing `pyproject.toml`, run `uv lock` and commit both files.

### Running the Pipeline
The extracted texts, models and results are part of the repository. To regenerate them:

```bash
uv run python scripts/pdf_converter.py    # corpora/*.pdf -> data/*.txt
uv run python src/preprocess.py           # data/*.txt -> models/ (about 2 minutes)
uv run python src/batch_analysis.py       # models/ -> results/
```

The training is deterministic: the same input produces identical models.

### Querying
```bash
uv run python src/query_tool.py "Innere Sicherheit"
uv run python src/query_tool.py "Wohnungsbau" --top 5
```

For each party, the tool prints the neighbours with their mean similarity and the number of seed models that rank them in their top 10.

### Dashboard
The dashboard loads `results/results.json` and therefore needs a web server. Locally:

```bash
uv run python -m http.server 8000         # then open http://localhost:8000
```

---

## Data Sources
The PDFs in [`corpora/`](corpora/) are the programs as published by the parties for the Bundestag election on 23 February 2025:

| File | Party | Title |
| :--- | :--- | :--- |
| `cdu.pdf` | CDU and CSU | Politikwechsel für Deutschland. Wahlprogramm von CDU und CSU |
| `gruene.pdf` | Bündnis 90/Die Grünen | Zusammen wachsen. Regierungsprogramm 2025 |
| `spd.pdf` | SPD | Mehr für Dich. Besser für Deutschland. Regierungsprogramm der SPD für die Bundestagswahl 2025 |
| `afd.pdf` | AfD | Zeit für Deutschland. Programm der Alternative für Deutschland für die Wahl zum 21. Deutschen Bundestag |

The code is licensed under the [MIT License](LICENSE). The election programs remain the property of the respective parties.
