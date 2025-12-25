from datasets import load_dataset
from collections import Counter, defaultdict

# Special tags to model the start and end of the sentence
START_TAG = "<START>"
END_TAG = "<END>"

# Special token to group rare words and handle Out-Of-Vocabulary (OOV) words during testing
UNKNOWN_WORD = "<UNK>"

def train_hmm_supervised_with_unk(
        dataset_name="batterydata/pos_tagging",
        split_name="train",
        word_field="words",
        tag_field="labels",
        min_word_frequency=2,
        laplace_smoothing=1.0
):

    # Objective: estimate parameters of a supervised HMM for POS-tagging.

    # Parameters to estimate:
    # - transition_probabilities[prev_tag][curr_tag] = P(curr_tag | prev_tag)
    # - emission_probabilities[tag][word]           = P(word | tag)

    # The dataset is supervised: every sentence already has (words, tags).
    # Therefore, Baum-Welch is NOT needed: we just count and normalize.

    # 1) LOAD DATASET
    # load_dataset downloads (if needed) and loads the dataset into cache.
    # dataset[split_name] gives you the "train" split (or "test", etc.)
    dataset = load_dataset(dataset_name)[split_name]

    # 2) FIRST PASS: COUNT WORD FREQUENCIES
    # Needed to decide which words are "rare" and must become <UNK>. Using Counter(), a special dictionary
    # where each item is: "key" : frequency
    word_frequency = Counter()

    # Iterate through all sentences in the training set
    for sentence in dataset:
        # Iterate through all words in the sentence
        for word in sentence[word_field]:
            # Increment word count
            word_frequency[word] += 1

    # Function applying the rule: if word is rare => <UNK>
    def normalize_word(word):

        # If a word appears fewer than min_word_frequency times in training,
        # map it to <UNK>.

        # Why? So during training I can estimate P(<UNK> | tag),
        # and in testing I can map unseen words to <UNK> instead of having prob 0.

        if word_frequency[word] < min_word_frequency:
            return UNKNOWN_WORD
        return word

    # 3) COUNTING STRUCTURES (counts)
    # These are the "sufficient statistics" of the supervised HMM.
    transition_count = Counter()
    # transition_count[(prev_tag, curr_tag)] = how many times we see prev_tag -> curr_tag

    emission_count = Counter()
    # emission_count[(tag, word)] = how many times the tag "tag" labels the word "word"

    tag_occurrence_count = Counter()
    # tag_occurrence_count[tag] = how many times that tag appears (needed to normalize emissions)

    previous_tag_count = Counter()
    # previous_tag_count[prev_tag] = how many times prev_tag appears as a "previous tag"
    # (needed to normalize transitions)

    # Sets we will use to know which tags and words exist in the model
    tag_set = {START_TAG, END_TAG}
    vocabulary = {UNKNOWN_WORD}

    # 4) SECOND PASS: SCANNING AND COUNTING TRANSITIONS/EMISSIONS
    for sentence in dataset:

        # Get words and apply normalization (<UNK> for rare words). Obtain a list of all words
        # in the dataset including UNKs
        words = [normalize_word(w) for w in sentence[word_field]]

        # Get tags (already provided by the dataset)
        tags = sentence[tag_field]

        # Consistency check: there must be a tag for every word
        if len(words) != len(tags):
            raise ValueError("Numero di parole e tag non coincide")

        # 4a) START TRANSITION: <START> -> first tag
        # This models the distribution of the first tag. For every sentence, START is the predecessor of the first word
        # (which, being first, wouldn't have predecessors otherwise). Needed to calculate P(t_1 | START).
        # Effectively: previous_tag_count[START_TAG] = number of sentences
        previous_tag_count[START_TAG] += 1

        # I am recording an observed transition from START to tags[0] (first element). This sets up
        # initial probabilities. Ex: counting START -> DT; if many sentences start with DT,
        # transition_count[(START_TAG, DT)] will be high.
        transition_count[(START_TAG, tags[0])] += 1

        # 4b) ITERATE POSITION BY POSITION IN THE SENTENCE
        for position, (word, tag) in enumerate(zip(words, tags)):

            # EMISSION: tag "tag" emits word "word". Used later to calculate P(w|t)
            emission_count[(tag, word)] += 1

            # Total count of the tag (denominator for emissions, i.e., Matrix B). Since: P(w|t) = c(t,w) / c(t)
            tag_occurrence_count[tag] += 1

            # Update tag and vocabulary sets. Will be used to calculate matrices A and B
            tag_set.add(tag)
            vocabulary.add(word)

            # INTERNAL TRANSITION: tag_{i-1} -> tag_i. Finding numerator and denominator to calculate:
            # P(t_i | t_i-1) = c(t_i-1, t_i) / c(t_i-1)
            if position > 0:                            # Start from pos > 0 because the initial value is estimated with START.
                previous_tag = tags[position - 1]
                transition_count[(previous_tag, tag)] += 1
                previous_tag_count[previous_tag] += 1

        # 4c) END TRANSITION: last tag -> <END>. Models how a sentence "ends". Same logic as START
        last_tag = tags[-1]
        transition_count[(last_tag, END_TAG)] += 1
        previous_tag_count[last_tag] += 1

    # 5) FROM FREQUENCIES TO PROBABILITIES: TRANSITIONS
    # We want: P(curr | prev) = (count(prev,curr) + k) / (count(prev) + k*|T|) because we use Laplace Smoothing.
    transition_probabilities = defaultdict(dict)

    number_of_tags = len(tag_set)

    for previous_tag in tag_set:
        # Denominator: how many times previous_tag was seen as predecessor
        # + smoothing over all possible destinations (|T|)
        denominator = previous_tag_count[previous_tag] + laplace_smoothing * number_of_tags

        for current_tag in tag_set:
            # Numerator: how many times we saw prev -> curr
            # + smoothing
            numerator = transition_count[(previous_tag, current_tag)] + laplace_smoothing

            transition_probabilities[previous_tag][current_tag] = numerator / denominator # Matrix A

    # 6) FROM FREQUENCIES TO PROBABILITIES: EMISSIONS
    # We want: P(word | tag) = (count(tag,word) + k) / (count(tag) + k*|V|)
    emission_probabilities = defaultdict(dict)

    vocabulary_size = len(vocabulary)

    for tag in tag_set:

        # START and END do not "emit words": they are just control states
        if tag in (START_TAG, END_TAG):
            continue

        # Denominator: how many times this tag appears
        # + smoothing over all vocabulary words (|V|)
        denominator = tag_occurrence_count[tag] + laplace_smoothing * vocabulary_size

        for word in vocabulary:
            # Numerator: how many times the tag emits that word
            # + smoothing
            numerator = emission_count[(tag, word)] + laplace_smoothing

            emission_probabilities[tag][word] = numerator / denominator # Matrix B

    # 7) RETURN THE MODEL
    # These are exactly the parameters you will use in Viterbi:
    # - transition_probabilities to move between tags
    # - emission_probabilities to evaluate how compatible a word is with a tag
    hmm_model = {
        "transition_probabilities": transition_probabilities,
        "emission_probabilities": emission_probabilities,
        "tags": sorted(tag_set),
        "vocabulary": sorted(vocabulary),
        "word_frequency": word_frequency
    }

    return hmm_model
