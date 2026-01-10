import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datasets import load_dataset
from viterbi import run_viterbi
# Note: Ensure 'hmm' and 'train' modules are available in your path or current directory

def plot_benchmark_results(token_acc, sent_acc, confusion_data, tags):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 1. Bar Chart of Accuracies ---
    metrics = ['Token Accuracy', 'Sentence Accuracy']
    values = [token_acc, sent_acc]
    sns.barplot(x=metrics, y=values, ax=ax1, palette="viridis")
    ax1.set_ylim(0, 105)
    ax1.set_title("General performance (%)", fontsize=14)
    for i, v in enumerate(values):
        ax1.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')

    # --- 2. Confusion Matrix (Top 10 Tags) ---
    # Filter the 10 most frequent tags to avoid cluttering the chart
    top_tags = [t for t, _ in Counter([gold for gold, pred in confusion_data]).most_common(10)]
    matrix = np.zeros((len(top_tags), len(top_tags)))
    tag_to_idx = {tag: i for i, tag in enumerate(top_tags)}

    for gold, pred in confusion_data:
        if gold in tag_to_idx and pred in tag_to_idx:
            matrix[tag_to_idx[gold]][tag_to_idx[pred]] += 1

    # Row normalization (percentages)
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
    print("Training model in progress...")
    # The model is trained by estimating transitions and emissions

    test_dataset = load_dataset(dataset_name)["test"]

    total_tokens, correct_tokens = 0, 0
    perfect_sentences, total_sentences = 0, len(test_dataset)
    confusion_pairs = []

    print(f"Analyzing {total_sentences} sentences...")
    for example in test_dataset:
        sentence = example["words"]
        true_tags = [str(t) for t in example["labels"]]

        # Running the Viterbi algorithm in log-space
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

    # Calculating metrics
    t_acc = (correct_tokens / total_tokens) * 100
    s_acc = (perfect_sentences / total_sentences) * 100

    print(f"\nBenchmark Completed.\nToken Accuracy: {t_acc:.2f}%\nSentence Accuracy: {s_acc:.2f}%")

    # Generating plots
    plot_benchmark_results(t_acc, s_acc, confusion_pairs, model["tags"])


def benchmark_pretrained_models(trained_models, alphas, dataset_name="batterydata/pos_tagging"):
    """
    Runs the benchmark on a list of already trained models.
    Visualizes the results on a Cartesian plane with a logarithmic X-axis.
    """
    print("Loading test dataset...")
    test_dataset = load_dataset(dataset_name)["test"]

    token_accs = []
    sent_accs = []

    for i, model in enumerate(trained_models):
        alpha = alphas[i]
        t_total, t_correct = 0, 0
        s_perfect = 0

        print(f"Evaluation in progress: Alpha = {alpha}...")

        for ex in test_dataset:
            words = ex["words"]
            gold_tags = [str(t) for t in ex["labels"]]

            try:
                # Run Viterbi
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

    # --- Plot Configuration ---
    plt.figure(figsize=(10, 6))

    # Plotting the two curves
    plt.plot(alphas, token_accs, marker='o', markersize=8, label='Token Accuracy', color='#2c3e50', linewidth=2)
    plt.plot(alphas, sent_accs, marker='s', markersize=8, label='Sentence Accuracy', color='#e74c3c', linewidth=2)

    # LOGARITHMIC SCALE
    plt.xscale('log')

    # Graphical embellishments
    plt.title('Model performance with different smoothing coefficient', fontsize=14, pad=15)
    plt.xlabel('Smoothing Coefficient ($l$) - Log Scale', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)  # Grid for both logarithmic levels
    plt.legend(frameon=True, loc='lower left')

    plt.tight_layout()
    plt.show()

    return token_accs, sent_accs