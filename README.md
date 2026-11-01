# 🔍 NLP Plagiarism & Paraphrase Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/Domain-Natural_Language_Processing-FF6F00?style=for-the-badge)
![Zero Dependency Core](https://img.shields.io/badge/Architecture-Zero_Dependency_Core-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

<p align="center">
  <b>A lightweight, high-performance Natural Language Processing engine that computes multi-metric lexical, semantic, and sequential text similarity using TF-IDF Vectorization, N-Gram Jaccard Index, and Dynamic Programming LCS.</b>
</p>

</div>

---

## 🌟 Overview

Detecting textual plagiarism requires more than simple exact string matching. Real-world plagiarism involves synonym substitutions, reordering of clauses, and sentence paraphrasing.

**NLP Plagiarism Detector** addresses this by blending three complementary NLP and algorithmic techniques into a single composite scoring pipeline:
1. **TF-IDF + Cosine Similarity:** Evaluates semantic and topical keyword distribution across corpus spaces.
2. **N-Gram Jaccard Coefficient:** Measures phrase-level and colocation overlap ($N=2, 3$).
3. **Longest Common Subsequence (LCS):** Uses Dynamic Programming to detect sequential copying and preserved syntactic flow despite minor word insertions or deletions.

---

## 🔬 Mathematical Formulations

### 1. TF-IDF Cosine Similarity
$$\text{Sim}_{\cos}(D_1, D_2) = \frac{\sum_{w \in D_1 \cap D_2} \text{TF}(w, D_1) \cdot \text{TF}(w, D_2) \cdot [\text{IDF}(w)]^2}{\| \vec{V}_1 \| \cdot \| \vec{V}_2 \|}$$

### 2. N-Gram Jaccard Index
$$J(S_1, S_2) = \frac{|S_1 \cap S_2|}{|S_1 \cup S_2|}$$
*(where $S_i$ represents the set of contiguous word $n$-grams)*

### 3. LCS Ratio (Dynamic Programming)
$$\text{LCS Ratio}(T_1, T_2) = \frac{2 \cdot \text{len}(\text{LCS}(T_1, T_2))}{|T_1| + |T_2|}$$

### 4. Composite Plagiarism Score
$$\text{Score} = 0.50 \cdot \text{TF-IDF} + 0.30 \cdot \text{Jaccard} + 0.20 \cdot \text{LCS}$$

---

## 📊 Qualitative Verdict Thresholds

| Composite Score | Severity Level | System Verdict |
| :---: | :---: | :--- |
| **75% – 100%** | 🚨 Critical | **Severe Plagiarism Detected** (Direct copying / near-duplicate) |
| **50% – 74%** | ⚠️ High | **Substantial Overlap / Paraphrasing Detected** |
| **25% – 49%** | ℹ️ Moderate | **Low-to-Moderate Overlap** (Common domain terminology) |
| **0% – 24%** | ✅ Safe | **Original Content** (Negligible similarity) |

---

## 📂 Project Structure

```
NLP_Plag_Detection_/
├── nlp_detector/
│   ├── __init__.py           # Package initializer
│   ├── preprocessor.py       # Stopword removal, regex cleaning, n-gram extraction
│   └── similarity.py         # TF-IDF, Cosine, Jaccard, & DP LCS engines
├── sample_docs/              # Benchmark document test suite
│   ├── original.txt          # Technical document on distributed consensus
│   ├── plagiarized.txt       # Direct plagiarized variant
│   ├── paraphrased.txt       # Paraphrased synonym replacement variant
│   └── unrelated.txt         # Completely independent text (culinary arts)
├── detector.py               # Standalone CLI execution tool
├── requirements.txt          # Optional dependencies
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 🛠️ Quick Start Guide

### 1. Prerequisites
- Python 3.8 or higher (No third-party packages required for the core engine!)

### 2. Clone the Repository
```bash
git clone https://github.com/Sankarshan-Mahamuni/NLP_Plag_Detection_.git
cd NLP_Plag_Detection_
```

### 3. Run the Built-in Test Suite
Run the automated test matrix comparing original against plagiarized, paraphrased, and unrelated documents:
```bash
python detector.py --demo
```

### 4. Compare Custom Documents
```bash
# Compare two specific files
python detector.py document_a.txt document_b.txt

# Batch scan all .txt files in a directory pairwise
python detector.py --batch sample_docs/
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
