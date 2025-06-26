import numpy as np
from neural_core.symbolic_phase_classifier import train_phase_classifier

if __name__ == "__main__":
    data_path = "symbolic_embeddings.npz"
    try:
        data = np.load(data_path)
        X = data["X"]
        y = data["y"]
        print(f"✅ Loaded {len(X)} embeddings from {data_path}")
        train_phase_classifier(X, y)
    except Exception as e:
        print(f"❌ Failed to load or train: {e}")
