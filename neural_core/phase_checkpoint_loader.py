import torch
import os
import numpy as np
from neural_core.symbolic_phase_classifier import PhaseClassifier, train_phase_classifier

CHECKPOINT_PATH = "neural_core/symbolic_phase_classifier.pt"

# === Loader ===
def load_phase_classifier(input_dim, auto_retrain=False):
    """
    Load the PhaseClassifier from disk,
    validate architecture alignment (input_dim),
    and prepare for reuse in visualizations, inference, or future integrations.

    If checkpoint is missing and auto_retrain=True, retrain using symbolic_embeddings.npz.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        if auto_retrain:
            print("⚠️ Checkpoint not found. Auto-retraining...")
            data = np.load("symbolic_embeddings.npz")
            X = data["X"]
            y = data["y"]
            train_phase_classifier(X, y)
        else:
            raise FileNotFoundError(f"❌ No checkpoint found at {CHECKPOINT_PATH}")

    model = PhaseClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    print("✅ PhaseClassifier checkpoint loaded successfully.")
    return model

# === Example usage ===
if __name__ == "__main__":
    data = np.load("symbolic_embeddings.npz")
    input_dim = data["X"].shape[1]
    model = load_phase_classifier(input_dim, auto_retrain=True)
    print(model)
