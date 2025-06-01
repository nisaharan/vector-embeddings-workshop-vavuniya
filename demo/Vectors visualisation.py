import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

glove_file = 'C:/Users/NGenhatharan/Desktop/glove.6B.100d.txt'


words = ["king", "queen", "man", "woman", "germany", "italy", "hitler", "mussolini", "will"]
embeddings = {}

with open(glove_file, 'r', encoding="utf8") as f:
    for line in f:
        tokens = line.strip().split()
        word = tokens[0]
        if word in words:
            embeddings[word] = np.array(tokens[1:], dtype='float32')
        if len(embeddings) == len(words):
            break

# Add analogies
embeddings["analogy1"] = embeddings["hitler"] + embeddings["germany"] - embeddings["italy"]
embeddings["analogy2"] = embeddings["king"] - embeddings["man"] + embeddings["woman"]

# PCA
labels = list(embeddings.keys())
X = np.vstack([embeddings[w] for w in labels])
X_2d = PCA(n_components=2).fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
for i, label in enumerate(labels):
    plt.scatter(X_2d[i, 0], X_2d[i, 1], color=colors[i], s=100)
    plt.text(X_2d[i, 0]+0.02, X_2d[i, 1], label, fontsize=12)

def draw_arrow(start, end, text):
    i, j = labels.index(start), labels.index(end)
    plt.arrow(X_2d[i, 0], X_2d[i, 1],
              X_2d[j, 0] - X_2d[i, 0],
              X_2d[j, 1] - X_2d[i, 1],
              head_width=0.05, length_includes_head=True, color='cyan', linestyle='--')
    plt.text((X_2d[i, 0] + X_2d[j, 0]) / 2,
             (X_2d[i, 1] + X_2d[j, 1]) / 2,
             text, color='cyan')

draw_arrow("hitler", "analogy1", "Hitler + Germany - Italy ≈ ?")
draw_arrow("king", "analogy2", "King - Man + Woman ≈ ?")

plt.title("Word Analogies with GloVe Embeddings (PCA 2D)")
plt.grid(True)
plt.tight_layout()
plt.show()
