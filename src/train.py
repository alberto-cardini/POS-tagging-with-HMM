from datasets import load_dataset
from collections import Counter, defaultdict

# Tag speciali per modellare l'inizio e la fine della frase
START_TAG = "<START>"
END_TAG = "<END>"

# Token speciale per raggruppare parole rare e gestire parole mai viste (OOV) in test
UNKNOWN_WORD = "<UNK>"

def train_hmm_supervised_with_unk(
        dataset_name="batterydata/pos_tagging",
        split_name="train",
        word_field="words",
        tag_field="labels",
        min_word_frequency=2,
        laplace_smoothing=1.0
):
    """
    Obiettivo: stimare i parametri di un HMM supervisionato per POS-tagging.

    Parametri da stimare:
    - transition_probabilities[prev_tag][curr_tag] = P(curr_tag | prev_tag)
    - emission_probabilities[tag][word]           = P(word | tag)

    Il dataset è supervisionato: ogni frase ha già (parole, tag).
    Quindi NON serve Baum-Welch: basta contare e normalizzare.
    """

    # 1) CARICAMENTO DATASET
    # load_dataset scarica (se serve) e carica il dataset in cache.
    # dataset[split_name] ti dà lo split "train" (o "test", ecc.)
    dataset = load_dataset(dataset_name)[split_name]

    # 2) PRIMO PASSAGGIO: CONTO FREQUENZE DELLE PAROLE
    # Serve per decidere quali parole sono "rare" e devono diventare <UNK>. Uso Counter() che è uno speciale dizionario
    # tale che ogni tupla è del tipo: "key" : frequency
    word_frequency = Counter()

    # Scorro tutte le frasi nel training set
    for sentence in dataset:
        # Scorro tutte le parole della frase
        for word in sentence[word_field]:
            # Incremento il conteggio della parola
            word_frequency[word] += 1

    # Funzione che applica la regola: se la parola è rara => <UNK>
    def normalize_word(word):
        """
        Se una parola compare meno di min_word_frequency volte nel training,
        la mappo a <UNK>.

        Perché? così durante il training posso stimare P(<UNK> | tag),
        e in test posso mappare parole mai viste a <UNK> invece di avere prob 0.
        """
        if word_frequency[word] < min_word_frequency:
            return UNKNOWN_WORD
        return word

    # 3) STRUTTURE DI CONTEGGIO (counts)
    # Questi sono i "sufficient statistics" dell'HMM supervisionato.
    transition_count = Counter()
    # transition_count[(prev_tag, curr_tag)] = quante volte vedo prev_tag -> curr_tag

    emission_count = Counter()
    # emission_count[(tag, word)] = quante volte il tag "tag" etichetta la parola "word"

    tag_occurrence_count = Counter()
    # tag_occurrence_count[tag] = quante volte compare quel tag (serve per normalizzare emissioni)

    previous_tag_count = Counter()
    # previous_tag_count[prev_tag] = quante volte prev_tag appare come "tag precedente"
    # (serve per normalizzare transizioni)

    # Insiemi che useremo per sapere quali tag e quali parole esistono nel modello
    tag_set = {START_TAG, END_TAG}
    vocabulary = {UNKNOWN_WORD}

    # 4) SECONDO PASSAGGIO: SCANSIONE E CONTEGGI DI TRANSIZIONI/EMISSIONI
    for sentence in dataset:

        # Prendo le parole e applico la normalizzazione (<UNK> per parole rare). Ottengo una lista di tutte le parole
        # del dataset e i vari UNK
        words = [normalize_word(w) for w in sentence[word_field]]

        # Prendo i tag (già forniti dal dataset)
        tags = sentence[tag_field]

        # Controllo di coerenza: deve esserci un tag per ogni parola
        if len(words) != len(tags):
            raise ValueError("Numero di parole e tag non coincide")

        # 4a) TRANSIZIONE DI INIZIO: <START> -> primo tag
        # Questo serve a modellare la distribuzione del primo tag. Per ogni frase START è il predecessore della prima parola
        # (che di per se, essendo la prima, non avrebbe predecessori e non potrei calcolare la prob di transizione). Mi
        # serve per calcolare P(t_1 | START). In pratica ho alla fine che: previous_tag_count[START_TAG] = numero frasi
        previous_tag_count[START_TAG] += 1

        # Sto imponendo di aver osservato una volta la transizione da START a tags[0] (primo elemento della frase). Lo
        # metto a tavolino per trovare le prob. iniziali. Es: sto contando START -> DT, se molte frasi cominciano con DT
        # allora transition_count[(START_TAG, DT)] sarà alto.
        transition_count[(START_TAG, tags[0])] += 1

        # 4b) SCORRO POSIZIONE PER POSIZIONE NELLA FRASE
        for position, (word, tag) in enumerate(zip(words, tags)):

            # EMISSIONE: il tag "tag" emette la parola "word". Lo uso poi per calcolare P(w|t)
            emission_count[(tag, word)] += 1

            # Conteggio totale del tag (denominatore per le emissioni, ovvero B). Visto che: P(w|t) = c(t,w) / c(t)
            tag_occurrence_count[tag] += 1

            # Aggiorno gli insiemi di tag e vocabolario. Lo userò per calcolare le matrici A e B
            tag_set.add(tag)
            vocabulary.add(word)

            # TRANSIZIONE INTERNA: tag_{i-1} -> tag_i. Qua sto trovando numeratore e denominatore che poi uso per calcolare:
            # P(t_i | t_i-1) = c(t_i-1, t_i) / c(t_i-1)
            if position > 0:                            # Parto da pos > 0 perchè il valore iniziale lo stimo con START.
                previous_tag = tags[position - 1]
                transition_count[(previous_tag, tag)] += 1
                previous_tag_count[previous_tag] += 1

        # 4c) TRANSIZIONE DI FINE: ultimo tag -> <END>. Serve a modellare come "termina" una frase. Stesso discorso dello START
        last_tag = tags[-1]
        transition_count[(last_tag, END_TAG)] += 1
        previous_tag_count[last_tag] += 1

    # 5) DALLE FREQUENZE ALLE PROBABILITÀ: TRANSIZIONI
    # Vogliamo: P(curr | prev) = (count(prev,curr) + k) / (count(prev) + k*|T|) Perchè uso Laplace Smoothing.
    transition_probabilities = defaultdict(dict)

    number_of_tags = len(tag_set)

    for previous_tag in tag_set:
        # Denominatore: quante volte ho visto previous_tag come predecessore
        # + smoothing su tutte le possibili destinazioni (|T|)
        denominator = previous_tag_count[previous_tag] + laplace_smoothing * number_of_tags

        for current_tag in tag_set:
            # Numeratore: quante volte ho visto prev -> curr
            # + smoothing
            numerator = transition_count[(previous_tag, current_tag)] + laplace_smoothing

            transition_probabilities[previous_tag][current_tag] = numerator / denominator # Matrice A

    # 6) DALLE FREQUENZE ALLE PROBABILITÀ: EMISSIONI
    # Vogliamo: P(word | tag) = (count(tag,word) + k) / (count(tag) + k*|V|)
    emission_probabilities = defaultdict(dict)

    vocabulary_size = len(vocabulary)

    for tag in tag_set:

        # START e END non "emettono parole": sono solo stati di controllo
        if tag in (START_TAG, END_TAG):
            continue

        # Denominatore: quante volte compare questo tag
        # + smoothing su tutte le parole del vocabolario (|V|)
        denominator = tag_occurrence_count[tag] + laplace_smoothing * vocabulary_size

        for word in vocabulary:
            # Numeratore: quante volte il tag emette quella parola
            # + smoothing
            numerator = emission_count[(tag, word)] + laplace_smoothing

            emission_probabilities[tag][word] = numerator / denominator # Matrice B

    # 7) RITORNO DEL MODELLO
    # Questi sono esattamente i parametri che userai in Viterbi:
    # - transition_probabilities per muoverti tra tag
    # - emission_probabilities per valutare quanto una parola è compatibile con un tag
    hmm_model = {
        "transition_probabilities": transition_probabilities,
        "emission_probabilities": emission_probabilities,
        "tags": sorted(tag_set),
        "vocabulary": sorted(vocabulary),
        "word_frequency": word_frequency
    }

    return hmm_model
