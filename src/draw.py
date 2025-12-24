import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_viterbi_heatmap(viterbi_matrix, tags, words, title="Viterbi Heatmap (Log-Prob)"):
    #Crea una heatmap pulita gestendo i valori -inf.
    plt.figure(figsize=(10, len(tags) * 0.4))

    # Sostituiamo -inf con un valore molto basso per non rompere la scala colori
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
    plt.title(title)
    plt.show()

def plot_transition_gradient(hmm_model, title= "HMM Transition Gradient (Sorted Log-Probs)"):
    """
    Crea una heatmap ordinando i tag per la somma delle loro log-probabilità.
    Non richiede SciPy.
    """
    tags = hmm_model["tags"]
    A = hmm_model["transition_probabilities"]

    # Calcoliamo la matrice densa
    n = len(tags)
    matrix_data = []

    for prev_tag in tags:
        row = [math.log(A[prev_tag].get(curr_tag, 1e-12)) for curr_tag in tags]
        matrix_data.append(row)

    matrix = np.array(matrix_data)

    # ORDINAMENTO PER GRADIENTE:
    # Calcoliamo la "forza" di ogni tag (somma delle probabilità di uscita)
    tag_strength = matrix.sum(axis=1)
    # Otteniamo gli indici che ordinano i tag dal più debole al più forte
    sorted_indices = np.argsort(tag_strength)

    # Riordiniamo la matrice e la lista dei tag
    sorted_tags = [tags[i] for i in sorted_indices]
    matrix = matrix[sorted_indices, :]  # Riordina righe
    matrix = matrix[:, sorted_indices]  # Riordina colonne

    # Visualizzazione
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
    Crea una heatmap Viridis e riquadra in rosso il percorso scelto.
    """
    plt.figure(figsize=(12, len(tags) * 0.4))

    # 1. Gestione dei valori -inf per non rovinare il gradiente
    clean_matrix = viterbi_matrix.copy()
    valid_values = clean_matrix[clean_matrix != -np.inf]
    floor_val = np.nanmin(valid_values) - 10 if valid_values.size > 0 else -100
    clean_matrix[clean_matrix == -np.inf] = floor_val

    # 2. Ordinamento per gradiente (opzionale, ma consigliato per estetica)
    row_means = clean_matrix.mean(axis=1)
    idx = np.argsort(row_means)
    sorted_matrix = clean_matrix[idx]
    sorted_tags = [tags[i] for i in idx]

    # Mappa per trovare rapidamente la nuova posizione Y di ogni tag dopo l'ordinamento
    tag_to_y = {tag: i for i, tag in enumerate(sorted_tags)}

    # 3. Disegno della Heatmap base
    ax = sns.heatmap(sorted_matrix, annot=True, fmt=".1f",
                     xticklabels=words, yticklabels=sorted_tags,
                     cmap="viridis", cbar_kws={'label': 'log-prob'})

    # 4. Aggiunta dei riquadri rossi per il best_path
    for col, chosen_tag in enumerate(best_path):
        if chosen_tag in tag_to_y:
            row = tag_to_y[chosen_tag]
            # Aggiungiamo un rettangolo rosso sulla cella (x, y)
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
    Crea una heatmap della matrice di transizione A.
    Mostra la probabilità P(Tag_corrente | Tag_precedente).
    """
    tags = hmm_model["tags"]
    n = len(tags)
    matrix = np.zeros((n, n))
    A = hmm_model["transition_probabilities"]

    # Riempiamo la matrice
    for i, prev_tag in enumerate(tags):
        for j, curr_tag in enumerate(tags):
            # Recuperiamo la probabilità (con un fallback minimo per evitare log(0))
            prob = A[prev_tag].get(curr_tag, 1e-12)
            matrix[i, j] = math.log(prob)

    # Configurazione estetica
    plt.figure(figsize=(12, 10))

    # Utilizziamo una colormap "calda" come rocket o magma per un look moderno
    sns.heatmap(matrix,
                annot=False,          # Setta a True se vuoi vedere i numeri dentro
                cmap="rocket",
                xticklabels=tags,
                yticklabels=tags,
                cbar_kws={'label': 'Log-Probability'})

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel("Tag Successivo ($t$)", fontsize=12)
    plt.ylabel("Tag Precedente ($t-1$)", fontsize=12)

    # Ruotiamo le etichette per leggibilità
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()