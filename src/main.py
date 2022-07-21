import os
import sys
import nltk
import json
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.decomposition import PCA

pca = PCA(3)
p = pipeline("feature-extraction", "gpt2", device=0, truncation=True) # edit model and device
colors = cycle('bgrcmk')
filenames = []

try:
    strip = nltk.data.load('tokenizers/punkt/english.pickle')
except:
    nltk.download('punkt')
    strip = nltk.data.load('tokenizers/punkt/english.pickle')


def get_vec(s):
    """Average hidden states into single n-dimensional vector."""
    return np.mean(p(s)[0], axis=0)


def reduce_dims(b):
    """Reduce all dimensions down to n dimensions with PCA."""
    v = pca.fit_transform([b[x]["v"] for x in b.keys()])
    for y, s in enumerate(b.keys()):
        b[s]["x"] = float(v[y][0])
        b[s]["y"] = float(v[y][1])
        b[s]["z"] = float(v[y][2])
        del b[s]["v"]
    return b


def load_works():
    """Load books, from cache if possible."""
    works = []
    for i in sys.argv[1:]:
        filenames.append(i)
        if f"{i}.json" in os.listdir("."):
            works.append(json.load(open(f"{i}.json", "r")))
        else:
            with open(f"{i}", "r") as f:
                s = strip.tokenize(" ".join(f.readlines()))
                # s = [" ".join(s[y:y+5]) for y in range(0, len(s), 5)]
                b = {y: {"s": x, "v": get_vec(x)} for y, x in enumerate(s)}
                b = reduce_dims(b)
                works.append(b)
                json.dump(b, open(f"{i}.json", "w"))
    return works


data = load_works()
vectors = []

for work in data: # pack together vectors
    vectors.append([[work[x]["x"],
                     work[x]["y"],
                     work[x]["z"]] for x in work.keys()])

fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection="3d")

for i, work in enumerate(vectors): # unpack vectors and display
    x = [x[0] for x in work] 
    y = [x[1] for x in work]
    z = [x[2] for x in work]
    ax.scatter3D(x, y, z, color=next(colors), label=filenames[i])

plt.legend()
plt.show()
