import os
import nltk
import json
import argparse
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.decomposition import PCA


def get_args():
    """Return arguments."""
    parser = argparse.ArgumentParser(description="Generate sentence" +
                                     " embeddings" +
                                     "for a set of texts and plot them in 3D.")
    parser.add_argument("-m", "--model",
                        help="valid Hugging Face model",
                        default="gpt2")
    parser.add_argument("-d", "--device",
                        type=int,
                        help="CUDA device ordinal, e.g. 0 or 1",
                        default=-1)
    parser.add_argument("-c", "--context",
                        help="generate sentence embeddings with prior context",
                        action="store_true")
    parser.add_argument("-o", "--opacity",
                        help="display embeddings with opacity per chronology",
                        action="store_true")
    parser.add_argument("-g", "--group",
                        type=int,
                        help="group sentences in n bins, defaults to 1",
                        default="1")
    parser.add_argument("-r", "--regenerate",
                        help="regenerate cache (use if `group` changes)",
                        action="store_true")
    parser.add_argument("texts",
                        nargs="*",
                        help="filenames separated by spaces")
    return parser.parse_args()


def get_vec(s, p):
    """Average hidden states into single n-dimensional vector."""
    return np.mean(p(s)[0], axis=0)


def reduce_dims(b):
    """Reduce all vectors down to n dimensions with PCA."""
    v = pca.fit_transform([b[x]["v"] for x in b.keys()])
    for y, s in enumerate(b.keys()):
        b[s]["x"] = float(v[y][0])
        b[s]["y"] = float(v[y][1])
        b[s]["z"] = float(v[y][2])
        del b[s]["v"]
    return b


def load_model():
    p = pipeline("feature-extraction",
                 args.model,
                 device=args.device,
                 truncation=True)
    return p


def load_works(files, p):
    """Load texts, from cache if possible."""
    works = []
    print(files)
    for i in files:
        filenames.append(i)
        if not args.regenerate and f"{i}.json" in os.listdir("."):
            print(f"Loading {i} from cache")
            works.append(json.load(open(f"{i}.json", "r")))
        else:
            if not p:
                p = load_model()
            print(f"Generating cache for {i}")
            with open(f"{i}", "r") as f:
                s = strip.tokenize(" ".join(f.readlines()))
                s = [" ".join(s[y:y+args.group]) for y in range(0,
                                                                len(s),
                                                                args.group)]
                b = {y: {"s": x, "v": get_vec(x, p)} for y, x in enumerate(s)}
                b = reduce_dims(b)
                works.append(b)
                json.dump(b, open(f"{i}.json", "w"))
    return works


if __name__ == "__main__":
    # load requisite objects
    args = get_args()
    pca = PCA(3)
    p = False
    colors = cycle([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0.5, 0.5, 0],
                    [0, 0.5, 0.5],
                    [0.5, 0, 0.5]])
    filenames = []
    try:
        strip = nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
        strip = nltk.data.load('tokenizers/punkt/english.pickle')

    # load texts and pack together vectors
    data = load_works(args.texts, p)
    vectors = []
    for work in data:
        vectors.append([[work[x]["x"],
                         work[x]["y"],
                         work[x]["z"]] for x in work.keys()])

    # display vectors
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection="3d")
    for i, work in enumerate(vectors):  # unpack vectors and display
        color = next(colors)
        if args.opacity:
            color = [color + [x/len(work)] for x in range(len(work))]
        else:
            color = [color for x in range(len(work))]
        x = [x[0] for x in work]
        y = [x[1] for x in work]
        z = [x[2] for x in work]
        ax.scatter3D(x, y, z,
                     c=color,
                     label=filenames[i])
    plt.legend()
    plt.show()
