from viterbi import *
import argparse
import os
import re
import time
from datetime import datetime
from train import *
from draw import plot_emission_probs, plot_viterbi_heatmap_with_path

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--sentence",
    type=str,
    default="all",
    help="ID frase: 1..5 oppure 'all'"
)
parser.add_argument(
    "-l", "--smoothing",
    type=float,
    default=None,
    help="Se specificato, allena solo quel modello. Se omesso: usa 1.0 e 0.001."
)
args = parser.parse_args()

sentences = {
    "1": ("Simple", ["John", "is", "going", "to", "the", "office", "."]),
    "2": ("No Period", ["John", "is", "going", "to", "the", "office"]),
    "3": ("Difficult", ["The", "complex", "houses", "many", "students", "."]),
    "4": ("No Sense", ["The", "gloopy", "zibber", "quops", "the", "flimflam", "."]),
    "5": ("No Sense Reverse", ["The", "quops", "flimflam", "gloopy", "the", "zibber", "."])
}

if args.sentence == "all":
    selected = list(sentences.items())
    run_name = "all_sentences"
else:
    if args.sentence not in sentences:
        raise ValueError(f"Invalid Sentence ID: {args.sentence}. Try 1..5 or 'all'.")
    selected = [(args.sentence, sentences[args.sentence])]
    run_name = f"sentence_{args.sentence}_{sentences[args.sentence][0]}"

run_name_slug = slugify(run_name)

out_root = os.path.join("plots", f"{run_name_slug}_{now_stamp()}")
os.makedirs(out_root, exist_ok=True)

print("Training phase...")

smoothings = [1.0, 0.001] if args.smoothing is None else [args.smoothing]
trained_models = []

for sm in smoothings:
    t0 = time.time()
    model = train_hmm_supervised_with_unk(smoothing=sm)
    t1 = time.time()
    print(f"Training (l={sm}) completed in: {t1 - t0:.4f} seconds")
    trained_models.append((sm, model))

for sid, (s_name, words) in selected:
    print(f"\n--- Analyzing Sentence {sid}: {s_name} ---")
    print(f"Sentence: {words}")

    sentence_slug = slugify(f"{sid}_{s_name}")

    for sm, model in trained_models:
        sm_label = str(sm).replace(".", "p")
        run_tag = f"l_{sm_label}"

        path, v_matrix, tags_list = run_viterbi(model, words)
        print(f"Predicted tags (l={sm}): {path}")

        out_dir = os.path.join(out_root, sentence_slug, run_tag)
        os.makedirs(out_dir, exist_ok=True)

        out1 = os.path.join(out_dir, f"{sentence_slug}_{run_tag}_emissions.png")
        plot_emission_probs(
            model,
            words,
            tags_list,
            title=f"{s_name} - Emission Probabilities (l = {sm})",
            save_path=out1,
            show=False,
            close=True
        )

        out2 = os.path.join(out_dir, f"{sentence_slug}_{run_tag}_viterbi_heatmap.png")
        plot_viterbi_heatmap_with_path(
            v_matrix,
            tags_list,
            words,
            path,
            title=f"{s_name} - Viterbi Heatmap (l = {sm})",
            save_path=out2,
            show=False,
            close=True
        )

print(f"Plots salvati in: {os.path.abspath(out_root)}")
