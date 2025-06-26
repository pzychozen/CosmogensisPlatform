import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

from neural_core.symbolic_phase_classifier import PhaseClassifier, auto_label_motion

LABELS = ["stable", "recursive", "chaotic"]

# === Load data and model ===
data = np.load("symbolic_embeddings.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
y_true = np.array([auto_label_motion(m) for m in data["y"]])

model = PhaseClassifier(input_dim=X.shape[1])
model.load_state_dict(torch.load("neural_core/symbolic_phase_classifier.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(X)
    y_pred = torch.argmax(logits, dim=1).numpy()

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(cmap="viridis")
plt.title("Phase Classifier — Confusion Matrix")
plt.tight_layout()
plt.show()

# === PCA Projection for Visualizing Class Separation ===
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.numpy())

plt.figure(figsize=(8,6))
for label_id, label_name in enumerate(LABELS):
    idx = y_pred == label_id
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label_name, alpha=0.6)

plt.legend()
plt.title("Phase Space — PCA Embedding Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()