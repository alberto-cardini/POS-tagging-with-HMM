from hmm import *

def evaluate_hmm(
    hmm_model,
    dataset_name="batterydata/pos_tagging",
    split_name="test",
    word_field="words",
    tag_field="labels"
):
    dataset = load_dataset(dataset_name)[split_name]

    correct = 0
    total = 0

    tag_confusion = Counter()

    for sentence in dataset:
        words = sentence[word_field]
        gold_tags = sentence[tag_field]

        pred_tags = run_viterbi(hmm_model, words)

        assert len(pred_tags) == len(gold_tags)

        for gold, pred in zip(gold_tags, pred_tags):
            total += 1
            if gold == pred:
                correct += 1
            tag_confusion[(gold, pred)] += 1

    accuracy = correct / total

    return accuracy, tag_confusion


# ======================
# MAIN
# ======================

if __name__ == "__main__":

    hmm_model = train_hmm_supervised_with_unk()

    accuracy, confusion = evaluate_hmm(hmm_model)

    print(f"\nToken-level accuracy: {accuracy:.4f}")

    print("\nMost common errors (gold → predicted):")
    for (gold, pred), count in confusion.most_common(10):
        if gold != pred:
            print(f"{gold} → {pred}: {count}")