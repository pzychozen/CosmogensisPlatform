import os
import json
import numpy as np
from scipy.spatial import KDTree

def load_fingerprint(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None

def analyze_fingerprints(folder):
    files = sorted([
        f for f in os.listdir(folder)
        if f.startswith("symbolic_fingerprint_") and f.endswith(".json")
    ], key=lambda x: int(''.join(filter(str.isdigit, x))))

    all_results = []
    prev_nodes = None

    for fname in files:
        full_path = os.path.join(folder, fname)
        data = load_fingerprint(full_path)
        if not data:
            continue

        for step in data:
            iter_num = step.get("iteration", None)
            nodes = np.array(step.get("nodes", []))
            if nodes.shape[0] < 2:
                continue

            motion = None
            if prev_nodes is not None and prev_nodes.shape == nodes.shape:
                try:
                    tree = KDTree(prev_nodes)
                    _, indices = tree.query(nodes)
                    diff = nodes - prev_nodes[indices]
                    motion = np.linalg.norm(diff, axis=1).mean()
                except:
                    motion = None

            result = {
                "iteration": iter_num,
                "node_count": nodes.shape[0],
                "avg_motion": round(motion, 3) if motion is not None else None
            }
            all_results.append(result)
            prev_nodes = nodes

    return all_results


def save_summary(results, out_path="symbolic_fingerprint_summary.csv"):
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "node_count", "avg_motion"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"✅ Summary saved to {out_path}")


if __name__ == "__main__":
    # 🔥 Change this to your full fingerprint path if needed
    fingerprint_dir = "data/fingerprint/run_1_26.06_09.27"

    results = analyze_fingerprints(fingerprint_dir)
    save_summary(results, out_path=os.path.join(fingerprint_dir, "summary.csv"))
