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
### Project Structure

* ```train.py```: Handles dataset loading and model training. It estimates transition and emission probabilities, managing Out-Of-Vocabulary words via the ```<UNK>``` token.

* ```viterbi.py```: Implements the Viterbi algorithm in log-space. It computes the most likely sequence of hidden states (tags) given an observation sequence.

* ```evaluation.py```: Contains benchmarking tools. It calculates token/sentence accuracy and generates confusion matrices and performance plots for different smoothing values.

* ```draw.py```: Visualization utilities using seaborn. It generates heatmaps for transition matrices, emission probabilities, and Viterbi path tracking.

* ```main.py```: Entry point of the project. Handles training, decoding, visualization, and plot saving through a simple command-line interface.

---

### Reproduce Report Results

The project is controlled via the `main.py` script. You can run it to train the model, decode specific sentences, and 
generate visualization plots in the `plots/` directory. Follow the steps in order to try the model:

#### 1. Setup

Install the required dependencies (preferably in a virtual environment) by running: ```pip install -r requirements.txt ```

#### 2. Running the Analysis
The script supports several command-line arguments:

* **The full execution (default)**: trains the model with smoothing values 1.0 and 0.001, then analyzes all pre-defined test 
sentences (Simple, Ambiguous, Nonsense/OOV). By running: 
     ``` bash
        python3 main.py
     ```  

* **Analyze Specific Sentences**: use the -s or --sentence flag to run specific test cases:

    (1) Simple, (2) No Period, (3) Difficult/Ambiguous, (4) Nonsense, (5) Nonsense Reverse 

    ```Bash
       # Example: Analyze the ambiguous sentence
       python main.py -s 3
    ``` 
    ```Bash
       # Example: Analyze the nonsense sentence to test OOV handling
       python main.py -s 4
    ``` 

* **Customize Smoothing**: use the -l or --smoothing flag to force a specific lambda value for smoothing.

    ``` Bash
        # Run every sentence with strong smoothing
        python main.py -l 0.5
    ``` 
    ``` Bash
        # Run on sentence (1) with strong smoothing
        python main.py -s 1 -l 0.5
    ``` 

* **Visualization**: after running the script, a plots/ directory is created containing detailed analytics for each run.

    - Emission Probability Plots: Shows the probability of the observed words being emitted by specific tags. 
    - Viterbi Heatmaps: visualizes the Viterbi matrix, highlighting the optimal path (sequence of tags) selected by the algorithm.

### 📄 Full Documentation

For mathematical details, benchmark results, and error analysis (including lexical ambiguity cases and nonsense word handling), please consult the full technical report:

<div align="center">

[![Download PDF](https://img.shields.io/badge/Download-Paper-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](./project-report.pdf)

</div>