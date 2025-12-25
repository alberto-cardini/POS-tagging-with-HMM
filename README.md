# Implementation of POS-tagging through Hidden Markov Model

This repository contains the implementation and theoretical analysis of a supervised **Hidden Markov Model (HMM)** for **Part-of-Speech (POS) Tagging**.

The project explores the effectiveness of the **Viterbi** algorithm in decoding grammatical sequences and analyzes the impact of different optimization techniques on model performance.

### Key Project Features
* **Probabilistic Model:** Utilization of HMM with first-order Markov assumptions.
* **Decoding:** Implementation of the Viterbi algorithm in log-space to prevent numerical underflow.
* **OOV Handling:** Strategies to manage *Out-Of-Vocabulary* (unknown) words using `<UNK>` tokens.
* **Smoothing:** Analysis of the impact of *Laplace smoothing* (additive) on transition and emission matrices.
* **Benchmark:** Evaluation of token-level and sentence-level accuracy on test datasets.

---

### 📄 Full Documentation

For mathematical details, benchmark results, and error analysis (including lexical ambiguity cases and nonsense word handling), please consult the full technical report:

<div align="center">

[![Download PDF](https://img.shields.io/badge/Download-Paper-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](./project-report.pdf)

</div>