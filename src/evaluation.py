import seaborn as sns
import numpy as np
from collections import Counter, defaultdict

def plot_benchmark_results(token_acc, sent_acc, confusion_data, tags):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 1. Grafico a Barre delle Accuratezze ---
    metrics = ['Token Accuracy', 'Sentence Accuracy']
    values = [token_acc, sent_acc]
    sns.barplot(x=metrics, y=values, ax=ax1, palette="viridis")
    ax1.set_ylim(0, 105)
    ax1.set_title("General performance (%)", fontsize=14)
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
    ax2.set_title("Confusion Matrix (% Top tag errors)", fontsize=14)
    ax2.set_xlabel("Inferred Tag")
    ax2.set_ylabel("Real Tag")

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


import matplotlib.pyplot as plt
from datasets import load_dataset
from hmm import run_viterbi


def benchmark_pretrained_models(trained_models, alphas, dataset_name="batterydata/pos_tagging"):
    """
    Esegue il benchmark su una lista di modelli già addestrati.
    Visualizza i risultati su un piano cartesiano con asse X logaritmico.
    """
    print("Caricamento dataset di test...")
    test_dataset = load_dataset(dataset_name)["test"]

    token_accs = []
    sent_accs = []

    for i, model in enumerate(trained_models):
        alpha = alphas[i]
        t_total, t_correct = 0, 0
        s_perfect = 0

        print(f"Valutazione in corso: Alpha = {alpha}...")

        for ex in test_dataset:
            words = ex["words"]
            gold_tags = [str(t) for t in ex["labels"]]

            try:
                # Esecuzione Viterbi
                pred_tags, _, _ = run_viterbi(model, words)

                matches = sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
                t_correct += matches
                t_total += len(gold_tags)
                if matches == len(gold_tags):
                    s_perfect += 1
            except Exception:
                continue

        token_accs.append((t_correct / t_total) * 100 if t_total > 0 else 0)
        sent_accs.append((s_perfect / len(test_dataset)) * 100)

    # --- Configurazione Plot ---
    plt.figure(figsize=(10, 6))

    # Plot delle due curve
    plt.plot(alphas, token_accs, marker='o', markersize=8, label='Token Accuracy', color='#2c3e50', linewidth=2)
    plt.plot(alphas, sent_accs, marker='s', markersize=8, label='Sentence Accuracy', color='#e74c3c', linewidth=2)

    # SCALA LOGARITMICA
    plt.xscale('log')

    # Abbellimenti grafici
    plt.title('Model performance with different smoothing coefficient', fontsize=14, pad=15)
    plt.xlabel('Smoothing Coefficient ($l$) - Log Scale', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)  # Grid per entrambi i livelli logaritmici
    plt.legend(frameon=True, loc='lower left')

    plt.tight_layout()
    plt.show()

    return token_accs, sent_accs


# Esempio di test rapido
if __name__ == "__main__":
    from train import train_hmm_supervised_with_unk

    # Array di test: copriamo diversi ordini di grandezza
    test_alphas = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100, 1000]
    print(f"Addestramento di {len(test_alphas)} modelli...")
    models = [train_hmm_supervised_with_unk(laplace_smoothing=a) for a in test_alphas]

    print(benchmark_pretrained_models(models, test_alphas))