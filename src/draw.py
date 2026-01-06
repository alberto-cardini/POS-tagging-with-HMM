import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def _finalize_plot(fig, save_path=None, show=True, close=True, dpi=200):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)


def plot_viterbi_heatmap(viterbi_matrix, tags, words,
                         title="Viterbi Heatmap (Log-Prob)",
                         save_path=None, show=True, close=True, dpi=200):
    # Creates a clean heatmap handling -inf values.
    fig = plt.figure(figsize=(10, len(tags) * 0.4))

    # Replace -inf with a very low value so as not to break the color scale
    clean_matrix = viterbi_matrix.copy()
    valid = clean_matrix[clean_matrix != -np.inf]
    floor_val = np.nanmin(valid) - 10 if valid.size > 0 else -100
    clean_matrix[clean_matrix == -np.inf] = floor_val

    sns.heatmap(clean_matrix, annot=True, fmt=".1f",
                xticklabels=words, yticklabels=tags,
                cmap="viridis", cbar_kws={'label': 'log-prob'})
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("POS Tags")

    _finalize_plot(fig, save_path, show, close, dpi)
    return fig


def plot_emission_probs(hmm_model, sentence, tags,
                        title="Sentence Emission Probabilities Matrix ( $P(Word | Tag)$ )",
                        save_path=None, show=True, close=True, dpi=200):
    # Shows how much each word 'attracts' the various tags (P(W|T)).
    words = sentence
    matrix = np.zeros((len(tags), len(words)))
    B = hmm_model["emission_probabilities"]

    for i, tag in enumerate(tags):
        for j, word in enumerate(words):
            # Using log for consistency
            prob = B[tag].get(word, B[tag].get("<UNK>", 1e-12))
            matrix[i, j] = math.log(prob)

    fig = plt.figure(figsize=(10, len(tags) * 0.4))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=words, yticklabels=tags)
    plt.title(title)

    _finalize_plot(fig, save_path, show, close, dpi)
    return fig


def plot_transition_gradient(hmm_model,
                             title="HMM Transition Gradient (Sorted Log-Probs)",
                             save_path=None, show=True, close=True, dpi=200):
    """
    Creates a heatmap sorting tags by the sum of their log-probabilities.
    Does not require SciPy.
    """
    tags = hmm_model["tags"]
    A = hmm_model["transition_probabilities"]

    # Compute the dense matrix
    matrix_data = []
    for prev_tag in tags:
        row = [math.log(A[prev_tag].get(curr_tag, 1e-12)) for curr_tag in tags]
        matrix_data.append(row)
    matrix = np.array(matrix_data)

    # GRADIENT SORTING:
    tag_strength = matrix.sum(axis=1)
    sorted_indices = np.argsort(tag_strength)

    sorted_tags = [tags[i] for i in sorted_indices]
    matrix = matrix[sorted_indices, :]
    matrix = matrix[:, sorted_indices]

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(matrix,
                cmap="magma",
                xticklabels=sorted_tags,
                yticklabels=sorted_tags,
                cbar_kws={'label': 'Log-Probability'})

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel("Current Tag ($t$)", fontsize=12)
    plt.ylabel("Previous Tag ($t-1$)", fontsize=12)

    _finalize_plot(fig, save_path, show, close, dpi)
    return fig


def plot_viterbi_heatmap_with_path(viterbi_matrix, tags, words, best_path,
                                   title="Viterbi Path Heatmap",
                                   save_path=None, show=True, close=True, dpi=200):
    """
    Creates a Viridis heatmap and outlines the chosen path in red.
    """
    fig = plt.figure(figsize=(12, len(tags) * 0.4))

    # 1. Handling -inf values to avoid ruining the gradient
    clean_matrix = viterbi_matrix.copy()
    valid_values = clean_matrix[clean_matrix != -np.inf]
    floor_val = np.nanmin(valid_values) - 10 if valid_values.size > 0 else -100
    clean_matrix[clean_matrix == -np.inf] = floor_val

    # 2. Gradient sorting (optional)
    row_means = clean_matrix.mean(axis=1)
    idx = np.argsort(row_means)
    sorted_matrix = clean_matrix[idx]
    sorted_tags = [tags[i] for i in idx]

    tag_to_y = {tag: i for i, tag in enumerate(sorted_tags)}

    ax = sns.heatmap(sorted_matrix, annot=True, fmt=".1f",
                     xticklabels=words, yticklabels=sorted_tags,
                     cmap="viridis", cbar_kws={'label': 'log-prob'})

    # 4. Adding red outlines for the best_path
    for col, chosen_tag in enumerate(best_path):
        if chosen_tag in tag_to_y:
            row = tag_to_y[chosen_tag]
            ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                       fill=False,
                                       edgecolor='red',
                                       lw=3,
                                       clip_on=False))

    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("POS Tags (Sorted)")

    _finalize_plot(fig, save_path, show, close, dpi)
    return fig


def plot_transition_heatmap(hmm_model,
                            title="Transition Probabilities (log-space)",
                            save_path=None, show=True, close=True, dpi=200):
    """
    Creates a heatmap of the transition matrix A.
    Shows the probability P(Current_Tag | Previous_Tag).
    """
    tags = hmm_model["tags"]
    n = len(tags)
    matrix = np.zeros((n, n))
    A = hmm_model["transition_probabilities"]

    for i, prev_tag in enumerate(tags):
        for j, curr_tag in enumerate(tags):
            prob = A[prev_tag].get(curr_tag, 1e-12)
            matrix[i, j] = math.log(prob)

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(matrix,
                annot=False,
                cmap="rocket",
                xticklabels=tags,
                yticklabels=tags,
                cbar_kws={'label': 'Log-Probability'})

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel("Next Tag ($t$)", fontsize=12)
    plt.ylabel("Previous Tag ($t-1$)", fontsize=12)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    _finalize_plot(fig, save_path, show, close, dpi)
    return fig
