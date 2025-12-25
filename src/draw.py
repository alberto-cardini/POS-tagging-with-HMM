import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_viterbi_heatmap(viterbi_matrix, tags, words, title="Viterbi Heatmap (Log-Prob)"):
    # Creates a clean heatmap handling -inf values.
    plt.figure(figsize=(10, len(tags) * 0.4))

    # Replace -inf with a very low value so as not to break the color scale
    clean_matrix = viterbi_matrix.copy()
    floor_val = np.nanmin(clean_matrix[clean_matrix != -np.inf]) - 10
    clean_matrix[clean_matrix == -np.inf] = floor_val

    sns.heatmap(clean_matrix, annot=True, fmt=".1f",
                xticklabels=words, yticklabels=tags,
                cmap="viridis", cbar_kws={'label': 'log-prob'})
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("POS Tags")
    plt.show()

def plot_emission_probs(hmm_model, sentence, tags, title = "Sentence Emission Probabilities Matrix ( $P(Word | Tag)$ )"):
    # Shows how much each word 'attracts' the various tags (P(W|T)).
    words = sentence
    matrix = np.zeros((len(tags), len(words)))
    B = hmm_model["emission_probabilities"]

    for i, tag in enumerate(tags):
        for j, word in enumerate(words):
            # Using log for consistency
            prob = B[tag].get(word, B[tag].get("<UNK>", 1e-12))
            matrix[i, j] = math.log(prob)

    plt.figure(figsize=(10, len(tags) * 0.4))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=words, yticklabels=tags)
    plt.title(title)
    plt.show()

def plot_transition_gradient(hmm_model, title= "HMM Transition Gradient (Sorted Log-Probs)"):
    """
    Creates a heatmap sorting tags by the sum of their log-probabilities.
    Does not require SciPy.
    """
    tags = hmm_model["tags"]
    A = hmm_model["transition_probabilities"]

    # Compute the dense matrix
    n = len(tags)
    matrix_data = []

    for prev_tag in tags:
        row = [math.log(A[prev_tag].get(curr_tag, 1e-12)) for curr_tag in tags]
        matrix_data.append(row)

    matrix = np.array(matrix_data)

    # GRADIENT SORTING:
    # Calculate the "strength" of each tag (sum of outgoing probabilities)
    tag_strength = matrix.sum(axis=1)
    # Get indices that sort tags from weakest to strongest
    sorted_indices = np.argsort(tag_strength)

    # Reorder the matrix and the tag list
    sorted_tags = [tags[i] for i in sorted_indices]
    matrix = matrix[sorted_indices, :]  # Reorder rows
    matrix = matrix[:, sorted_indices]  # Reorder columns

    # Visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix,
                cmap="magma",
                xticklabels=sorted_tags,
                yticklabels=sorted_tags,
                cbar_kws={'label': 'Log-Probability'})

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel("Current Tag ($t$)", fontsize=12)
    plt.ylabel("Previous Tag ($t-1$)", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_viterbi_heatmap_with_path(viterbi_matrix, tags, words, best_path, title="Viterbi Path Heatmap"):
    """
    Creates a Viridis heatmap and outlines the chosen path in red.
    """
    plt.figure(figsize=(12, len(tags) * 0.4))

    # 1. Handling -inf values to avoid ruining the gradient
    clean_matrix = viterbi_matrix.copy()
    valid_values = clean_matrix[clean_matrix != -np.inf]
    floor_val = np.nanmin(valid_values) - 10 if valid_values.size > 0 else -100
    clean_matrix[clean_matrix == -np.inf] = floor_val

    # 2. Gradient sorting (optional, but recommended for aesthetics)
    row_means = clean_matrix.mean(axis=1)
    idx = np.argsort(row_means)
    sorted_matrix = clean_matrix[idx]
    sorted_tags = [tags[i] for i in idx]

    # Map to quickly find the new Y position of each tag after sorting
    tag_to_y = {tag: i for i, tag in enumerate(sorted_tags)}

    # 3. Drawing the base Heatmap
    ax = sns.heatmap(sorted_matrix, annot=True, fmt=".1f",
                     xticklabels=words, yticklabels=sorted_tags,
                     cmap="viridis", cbar_kws={'label': 'log-prob'})

    # 4. Adding red outlines for the best_path
    for col, chosen_tag in enumerate(best_path):
        if chosen_tag in tag_to_y:
            row = tag_to_y[chosen_tag]
            # Add a red rectangle on cell (x, y)
            ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                       fill=False,
                                       edgecolor='red',
                                       lw=3,
                                       clip_on=False))

    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("POS Tags (Sorted)")
    plt.show()

def plot_transition_heatmap(hmm_model, title="Transition Probabilities (log-space)"):
    """
    Creates a heatmap of the transition matrix A.
    Shows the probability P(Current_Tag | Previous_Tag).
    """
    tags = hmm_model["tags"]
    n = len(tags)
    matrix = np.zeros((n, n))
    A = hmm_model["transition_probabilities"]

    # Fill the matrix
    for i, prev_tag in enumerate(tags):
        for j, curr_tag in enumerate(tags):
            # Retrieve probability (with a minimal fallback to avoid log(0))
            prob = A[prev_tag].get(curr_tag, 1e-12)
            matrix[i, j] = math.log(prob)

    # Aesthetic configuration
    plt.figure(figsize=(12, 10))

    # Use a "warm" colormap like rocket or magma for a modern look
    sns.heatmap(matrix,
                annot=False,          # Set to True if you want to see numbers inside
                cmap="rocket",
                xticklabels=tags,
                yticklabels=tags,
                cbar_kws={'label': 'Log-Probability'})

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel("Next Tag ($t$)", fontsize=12)
    plt.ylabel("Previous Tag ($t-1$)", fontsize=12)

    # Rotate labels for readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()