import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from datasets import load_dataset

# Importiamo le funzioni dai tuoi file
from train import train_hmm_supervised_with_unk
from hmm import run_viterbi

def plot_benchmark_results(token_acc, sent_acc, confusion_data, tags):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 1. Grafico a Barre delle Accuratezze ---
    metrics = ['Token Accuracy', 'Sentence Accuracy']
    values = [token_acc, sent_acc]
    sns.barplot(x=metrics, y=values, ax=ax1, palette="viridis")
    ax1.set_ylim(0, 105)
    ax1.set_title("Performance Generali (%)", fontsize=14)
    for i, v in enumerate(values):
        ax1.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')

    # --- 2. Matrice di Confusione (Top 10 Tag) ---
    # Filtriamo i 10 tag più frequenti per non affollare il grafico
    top_tags = [t for t, _ in Counter([gold for gold, pred in confusion_data]).most_common(10)]
    matrix = np.zeros((len(top_tags), len(top_tags)))
    tag_to_idx = {tag: i for i, tag in enumerate(top_tags)}

    for gold, pred in confusion_data:
        if gold in tag_to_idx and pred in tag_to_idx:
            matrix[tag_to_idx[gold]][tag_to_idx[pred]] += 1

    # Normalizzazione per righe (percentuali)
    row_sums = matrix.sum(axis=1)
    matrix_perc = np.divide(matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0) * 100

    sns.heatmap(matrix_perc, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=top_tags, yticklabels=top_tags, ax=ax2)
    ax2.set_title("Matrice di Confusione (% errori tra Top Tag)", fontsize=14)
    ax2.set_xlabel("Tag Predetto")
    ax2.set_ylabel("Tag Reale")

    plt.tight_layout()
    plt.show()


def run_benchmark(model, dataset_name= "batterydata/pos_tagging"):
    print("Addestramento modello in corso...")
    # Il modello viene addestrato stimando transizioni ed emissioni

    test_dataset = load_dataset(dataset_name)["test"]

    total_tokens, correct_tokens = 0, 0
    perfect_sentences, total_sentences = 0, len(test_dataset)
    confusion_pairs = []

    print(f"Analisi di {total_sentences} frasi...")
    for example in test_dataset:
        sentence = example["words"]
        true_tags = [str(t) for t in example["labels"]]

        # Esecuzione dell'algoritmo di Viterbi in log-space
        try:
            predicted_path, _, _ = run_viterbi(model, sentence)
        except:
            continue

        is_perfect = True
        for gold, pred in zip(true_tags, predicted_path):
            total_tokens += 1
            confusion_pairs.append((gold, pred))
            if gold == pred:
                correct_tokens += 1
            else:
                is_perfect = False

        if is_perfect:
            perfect_sentences += 1

    # Calcolo metriche
    t_acc = (correct_tokens / total_tokens) * 100
    s_acc = (perfect_sentences / total_sentences) * 100

    print(f"\nBenchmark Completato.\nAccuratezza Token: {t_acc:.2f}%\nAccuratezza Frase: {s_acc:.2f}%")

    # Generazione dei grafici
    plot_benchmark_results(t_acc, s_acc, confusion_pairs, model["tags"])