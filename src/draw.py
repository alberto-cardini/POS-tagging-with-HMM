import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

def plot_viterbi_heatmap(viterbi_matrix, tags, words, title="Viterbi Heatmap (Log-Prob)"):
    #Crea una heatmap pulita gestendo i valori -inf.
    plt.figure(figsize=(12, len(tags) * 0.4))

    # Sostituiamo -inf con un valore molto basso per non rompere la scala colori
    clean_matrix = viterbi_matrix.copy()
    floor_val = np.nanmin(clean_matrix[clean_matrix != -np.inf]) - 10
    clean_matrix[clean_matrix == -np.inf] = floor_val

    sns.heatmap(clean_matrix, annot=True, fmt=".1f",
                xticklabels=words, yticklabels=tags,
                cmap="viridis", cbar_kws={'label': 'log-prob'})
    plt.title(title)
    plt.xlabel("Parole")
    plt.ylabel("POS Tags")
    plt.show()


def plot_viterbi_3d(viterbi_matrix, tags, words):

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Prepariamo le coordinate
    x_pos, y_pos = np.meshgrid(np.arange(len(words)), np.arange(len(tags)))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)

    # Valori Z (altezze): usiamo un offset per rendere i log-probs positivi
    # Più il valore è vicino a 0 (alto), più la barra è alta
    z_vals = viterbi_matrix.flatten()
    floor_val = np.nanmin(z_vals[z_vals != -np.inf])
    z_vals = [val - floor_val if val != -np.inf else 0 for val in z_vals]

    dx = 0.5
    dy = 0.2
    dz = z_vals

    # Colore basato sull'altezza
    colors = plt.cm.plasma(np.array(dz) / max(dz) if max(dz) != 0 else 1)

    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=8)

    # 4. Aumenta la distanza delle etichette dall'asse
    ax.yaxis.labelpad = 20

    ax.set_xticks(np.arange(len(words)) + 0.75)
    ax.set_xticklabels(words, rotation=45)
    ax.set_yticks(np.arange(len(tags)) + 0.25)
    ax.set_yticklabels(tags)
    ax.set_zlabel("Confidenza (Log-Likelihood)")
    plt.title("Analisi 3D delle probabilità di Viterbi")
    plt.show()


def plot_emission_probs(hmm_model, sentence, tags):
    # Mostra quanto ogni parola 'attira' i vari tag (P(W|T)).
    words = sentence
    matrix = np.zeros((len(tags), len(words)))
    B = hmm_model["emission_probabilities"]

    for i, tag in enumerate(tags):
        for j, word in enumerate(words):
            # Usiamo log per coerenza
            prob = B[tag].get(word, B[tag].get("<UNK>", 1e-12))
            matrix[i, j] = math.log(prob)

    plt.figure(figsize=(10, len(tags) * 0.4))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=words, yticklabels=tags)
    plt.title("Sentence Emission Probabilities Matrix ( P(Word | Tag) )")
    plt.show()

def draw_viterbi_trellis_with_path(hmm_model, sentence, best_path, figsize_mult=(1.8, 0.45)):

    B = hmm_model["emission_probabilities"]
    # Escludiamo i tag tecnici come fatto nel tuo codice viterbi
    tags = [t for t in hmm_model["tags"] if t not in ("<START>", "<END>")]

    n_words = len(sentence)
    n_tags = len(tags)

    # Dimensionamento dinamico per evitare sovrapposizioni
    fig, ax = plt.subplots(figsize=(n_words * figsize_mult[0], n_tags * figsize_mult[1]))

    # Dizionario per mappare Tag -> Posizione Y nel grafico
    # Usiamo reversed(tags) per avere i tag in ordine alfabetico dall'alto al basso
    tag_to_y = {tag: i for i, tag in enumerate(reversed(tags))}

    # 1. DISEGNO DEI NODI (EMISSIONI)
    for col, word in enumerate(sentence):
        probs = [B[tag].get(word, B[tag].get("<UNK>", 1e-12)) for tag in tags]
        max_p, min_p = max(probs), min(probs)

        for tag in tags:
            row = tag_to_y[tag]
            p = B[tag].get(word, B[tag].get("<UNK>", 1e-12))

            # Scala logaritmica per l'intensità del colore
            intensity = (np.log(p) - np.log(min_p)) / (np.log(max_p) - np.log(min_p) + 1e-9)

            # Il nodo del path è evidenziato con un bordo più scuro
            is_in_path = (best_path[col] == tag)
            color = plt.cm.Blues(0.1 + 0.8 * intensity)

            ellipse = patches.Ellipse((col, row), 0.75, 0.65,
                                      edgecolor="black" if is_in_path else "silver",
                                      facecolor=color,
                                      linewidth=2.5 if is_in_path else 1,
                                      zorder=3)
            ax.add_patch(ellipse)

            # Testo del Tag
            ax.text(col, row, tag, ha='center', va='center',
                    fontsize=8, fontweight='bold' if is_in_path else 'normal',
                    color="white" if intensity > 0.5 else "black", zorder=4)

    for t in range(n_words - 1):
        start_node = (t, tag_to_y[best_path[t]])
        end_node = (t + 1, tag_to_y[best_path[t + 1]])

        ax.annotate("",
                    xy=end_node, xycoords='data',
                    xytext=start_node, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=3, shrinkA=15, shrinkB=15,
                                    connectionstyle="arc3,rad=0.1"),
                    zorder=2)

    ax.set_xlim(-0.7, n_words - 0.3)
    ax.set_ylim(-0.7, n_tags - 0.3)
    ax.set_xticks(range(n_words))
    ax.set_xticklabels(sentence, fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_axis_off()
    plt.title(f"Viterbi Path: {' -> '.join(best_path)}", pad=20)

    plt.tight_layout()
    plt.show()
