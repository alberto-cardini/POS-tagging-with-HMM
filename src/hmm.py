from draw import *

# --- LOGICA DI CALCOLO (VITERBI) ---

def run_viterbi(hmm_model, sentence):
    """
    Esegue l'algoritmo di Viterbi in log-space.
    Ritorna: best_path (lista), viterbi_matrix (numpy array), tags (lista)
    """
    A = hmm_model["transition_probabilities"]
    B = hmm_model["emission_probabilities"]
    # Escludiamo i tag di controllo per la matrice di visualizzazione
    tags = [t for t in hmm_model["tags"] if t not in ("<START>", "<END>")]
    vocab = set(hmm_model["vocabulary"])

    n = len(sentence)
    words = [w if w in vocab else "<UNK>" for w in sentence]

    # Inizializziamo la tabella con -inf (log-space)
    # Righe = Tags, Colonne = Parole della frase
    viterbi_table = np.full((len(tags), n), -np.inf)
    backpointer = np.zeros((len(tags), n), dtype=int)

    # 1. Inizializzazione (Giorno 0)
    for i, tag in enumerate(tags):
        trans_prob = A["<START>"].get(tag, 1e-12)
        emis_prob = B[tag].get(words[0], B[tag].get("<UNK>", 1e-12))
        viterbi_table[i, 0] = math.log(trans_prob) + math.log(emis_prob)

    # 2. Ricorsione
    for t in range(1, n):
        for i, curr_tag in enumerate(tags):
            emis_prob = B[curr_tag].get(words[t], B[curr_tag].get("<UNK>", 1e-12))

            # Calcolo vettoriale delle probabilità dai tag precedenti
            # viterbi_table[:, t-1] sono i log-probs del passo precedente
            # trans_probs sono i log-probs di transizione da tutti i prev_tags a curr_tag
            trans_probs = np.array([math.log(A[prev_tag].get(curr_tag, 1e-12)) for prev_tag in tags])

            probs = viterbi_table[:, t - 1] + trans_probs + math.log(emis_prob)

            viterbi_table[i, t] = np.max(probs)
            backpointer[i, t] = np.argmax(probs)

    # 3. Terminazione (Verso <END>)
    last_probs = np.array([viterbi_table[i, n - 1] + math.log(A[tag].get("<END>", 1e-12))
                           for i, tag in enumerate(tags)])
    best_last_idx = np.argmax(last_probs)

    # 4. Backtracking
    best_path_indices = [0] * n
    best_path_indices[-1] = best_last_idx
    for t in range(n - 1, 0, -1):
        best_path_indices[t - 1] = backpointer[best_path_indices[t], t]

    best_path = [tags[i] for i in best_path_indices]

    return best_path, viterbi_table, tags

# --- MAIN ---

if __name__ == "__main__":
    from train import *

    print("Training in corso...")
    model = train_hmm_supervised_with_unk()

    test_sentence = ["I", "saw", "her", "duck", "."]
    path, v_matrix, tags_list = run_viterbi(model, test_sentence)
    print("Tag stimati:", path)

    # Visualizzazioni
    plot_viterbi_heatmap(v_matrix, tags_list, test_sentence)
    plot_emission_probs(model, test_sentence, tags_list)