from draw import *

def run_viterbi(hmm_model, sentence):
    """
    Esegue l'algoritmo di Viterbi in log-space.
    Ritorna: best_path (lista), viterbi_matrix (numpy array), tags (lista)
    """

    # Raccolgo i parametri risultati dall'apprendimento sul dataset
    A = hmm_model["transition_probabilities"]
    B = hmm_model["emission_probabilities"]

    # Escludiamo i tag di controllo per la matrice di visualizzazione.
    tags = [t for t in hmm_model["tags"] if t not in ("<START>", "<END>")]
    vocab = set(hmm_model["vocabulary"])

    n = len(sentence)

    # Scorro tutte le parola della frase input e taggo quelle sconosciute come UNK
    words = [w if w in vocab else "<UNK>" for w in sentence]

    # Inizializziamo la tabella con -inf (log-space). Ovvero metto tutto a probabilità zero
    # Righe = Tags, Colonne = Parole della frase
    viterbi_table = np.full((len(tags), n), -np.inf)
    backpointer = np.zeros((len(tags), n), dtype=int)

    # 1. Inizializzazione
    for i, tag in enumerate(tags):
        # Probabilità che il tag X inizi la frase. Quindi prendo la P(X | <START>)
        trans_prob = A["<START>"].get(tag, 1e-12)

        # Probabilità che il tag X generi la prima osservazione, vale a dire, la prima parola della frase input
        emis_prob = B[tag].get(words[0], B[tag].get("<UNK>", 1e-12))

        viterbi_table[i, 0] = math.log(trans_prob) + math.log(emis_prob)

    # 2. Ricorsione
    for t in range(1, n):
        for i, curr_tag in enumerate(tags):

            # b_s(o_t)
            emis_prob = B[curr_tag].get(words[t], B[curr_tag].get("<UNK>", 1e-12))

            # Calcolo vettoriale delle probabilità dai tag precedenti
            # viterbi_table[:, t-1] sono i log-probs del passo precedente
            # trans_probs sono i log-probs di transizione da tutti i prev_tags a curr_tag
            trans_probs = np.array([math.log(A[prev_tag].get(curr_tag, 1e-12)) for prev_tag in tags])

            probs = viterbi_table[:, t - 1] + trans_probs + math.log(emis_prob)

            viterbi_table[i, t] = np.max(probs)
            backpointer[i, t] = np.argmax(probs)

    # 3. Terminazione
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

if __name__ == "__main__":
    from train import *

    print("Training in corso...")

    model_laplace = train_hmm_supervised_with_unk()
    model_001 = train_hmm_supervised_with_unk(laplace_smoothing = 0.001)

    test_sentence = ["John", "is", "going", "to", "the", "office", "."]

    path_laplace, v_matrix_laplace, tags_list_laplace = run_viterbi(model_laplace, test_sentence)
    print("Tag stimati con l = 1:", path_laplace)

    path_001, v_matrix_001, tags_list_001 = run_viterbi(model_001, test_sentence)
    print("Tag stimati con l = 0.001:", path_001)

    # Visualizzazioni
    plot_emission_probs(model_laplace, test_sentence, tags_list_laplace, title = "Sentence Emission Probabilities Matrix ( l = 1 )")
    plot_viterbi_heatmap_with_path(v_matrix_laplace, tags_list_laplace, test_sentence, path_laplace, title="Viterbi Heatmap ( l = 1 )")

    plot_emission_probs(model_001, test_sentence, tags_list_001, title="Sentence Emission Probabilities Matrix ( l = 0.001 )")
    plot_viterbi_heatmap_with_path(v_matrix_001, tags_list_001, test_sentence, path_001, title="Viterbi Heatmap ( l = 0.001 )")
