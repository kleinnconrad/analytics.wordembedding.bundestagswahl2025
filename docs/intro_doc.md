# Meaning Through Context

## Table of Contents
* [Overview](#overview)
* [The Core Idea: Context Over Word Counts](#the-core-idea-context-over-word-counts)
* [Methodology: Discovering Meaning through Vectors](#methodology-discovering-meaning-through-vectors)
* [The Analytical Pipeline](#the-analytical-pipeline)

## Overview
This repository applies word embeddings to political language. When analyzing election programs, the objective is to go beyond keyword counts and to examine **in which context a party uses a term**. The pipeline compares the election programs of CDU/CSU, Grüne, SPD and AfD for the 2025 Bundestag election.

---

## The Core Idea: Context Over Word Counts
Political text analysis often relies on curated keyword lists or frequency counts (e.g. "How often does the CDU mention 'Wirtschaft'?"). A count shows that a term is used, but not in which context, and the keyword list depends on the assumptions of the analyst.

Word embeddings address two questions:
* Which terms does a program use in the same contexts as a given term, including terms an analyst would not have searched for?
* How do these contexts differ between the parties?

### The "Anti-Thesaurus" Principle
The model is not a thesaurus. Its neighbours are not synonyms but words that occur in similar contexts. This is the intended behaviour: when two parties use the same word, the neighbours show the different contexts. For "Klimaschutz", the five nearest neighbours are:
* **SPD:** verfolgen, künstliche, erneuerbar, erhalt, nachhaltigkeit
* **AfD:** filmförderung, verankern, vermeintlich, name, klima

The neighbours are indications that have to be checked in the source text. Several of them are only interpretable together with the sentences in which they occur.

---

## Methodology: Discovering Meaning through Vectors
A separate skip-gram Word2Vec model (a shallow neural network) is trained on each party's program.

1. **Vector generation:** The model assigns a 100-dimensional vector to every word that occurs at least three times in the program. The vector is learned by predicting the surrounding words of each occurrence.
2. **Contextual similarity:** The cosine of the angle between two vectors measures how similar the contexts of the two words are. The smaller the angle, the more similar the contexts.

---

## The Analytical Pipeline

### 1. Training One Model per Party
Each program is trained separately (CDU/CSU, Grüne, SPD, AfD), so that every model reflects only the language of one program. Before training, the text is split into sentences, lemmatized, and stop words are removed. Each party is trained with five random seeds and identical parameters.

### 2. Identification
The query tool and the dashboard return the nearest neighbours of a term (e.g. "Innere Sicherheit") in each model, averaged over the five seed models.

### 3. Evaluation and Interpretation
Two measures indicate how reliable a neighbour is: the mean cosine similarity and the number of seed models that return it. The corpora are small (12,000 to 23,000 tokens), so the results are exploratory: they point to differences in the use of terms that can then be verified in the programs. They do not replace reading the programs.
