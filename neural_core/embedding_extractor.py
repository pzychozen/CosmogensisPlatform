import os
import torch
import json
import numpy as np
from tqdm import tqdm

from neural_core.gnn_loader import fingerprint_to_graphs
from neural_core.symbolic_transformer import SymbolicTransformer

# === Configuration ===
FOLDER = "data/fingerprint"
MAX_FILES = 1000  # safety cap to avoid memory overload
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json_fingerprints(folder):
    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # sort by number
    return [os.path.join(folder, f) for f in json_files[:MAX_FILES]]

def extract_embeddings():
    model = SymbolicTransformer().to(DEVICE)
    model.load_state_dict(torch.load("symbolic_transformer.pt", map_location=DEVICE))
    model.eval()

    embedding_list = []
    motion_targets = []

    files = load_json_fingerprints(FOLDER)
    print(f"🔍 Processing {len(files)} .json fingerprint files...")

    for filepath in tqdm(files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            for snapshot in data:
                iteration = snapshot.get("iteration", 0)
                motion = snapshot.get("avg_motion", None)
                if motion is None:
                    continue

                graphs = fingerprint_to_graphs([snapshot])
                if not graphs:
                    continue

                g = graphs[0].to(DEVICE)
                with torch.no_grad():
                    latent = model.encoder(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long).to(DEVICE))
                    embedding_list.append(latent.cpu().numpy())
                    motion_targets.append(motion)
        
        except Exception as e:
            print(f"[ERROR] Failed on {filepath}: {e}")

    print(f"✅ Collected {len(embedding_list)} embeddings.")
    return np.array(embedding_list), np.array(motion_targets)

if __name__ == "__main__":
    X, y = extract_embeddings()
    np.savez("symbolic_embeddings.npz", X=X, y=y)
    print("📁 Saved embeddings to symbolic_embeddings.npz")
