import itertools
import csv
import os
import numpy as np
from datetime import datetime
from engine import RecursiveUniverse
from utils import create_run_dirs

# === Main Scan Runner ===
def run_scan(params_grid, steps_per_run=100, run_id=1):
    dirs = create_run_dirs(run_id)
    output_file = os.path.join(dirs["analyzis"], "latest_run_results.csv")

    params_keys = list(params_grid.keys())
    param_combos = list(itertools.product(*params_grid.values()))

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(params_keys + ["iteration", "node_count", "avg_motion"])

        for values in param_combos:
            param_dict = dict(zip(params_keys, values))
            print(f"\n⚙️ Running: {param_dict}")

            universe = RecursiveUniverse(
                use_vesica=True,
                vesica_strength=param_dict.get("vesica_strength", 0.4),
                params={"eta": param_dict.get("eta", 0.8)},
            )

            for _ in range(steps_per_run):
                universe.step()

            mem = universe.memory.memory
            if not mem:
                print("❌ No valid memory entries. Skipping...")
                continue

            last = mem[-1]
            centroids = last.get("centroids", None)
            if not isinstance(centroids, np.ndarray) or centroids.ndim != 2:
                print("⚠️ Skipping due to malformed centroids.")
                continue

            node_count = len(centroids)

            motions = universe.memory.compute_motion_vectors()
            if motions:
                avg_motion = np.mean([m["magnitudes"].mean() for m in motions if m["magnitudes"].size > 0])
            else:
                avg_motion = 0.0

            writer.writerow(list(values) + [universe.iteration, node_count, avg_motion])

            # Export fingerprint JSON (optional)
            safe_id = "_".join(map(str, values))
            outpath = os.path.join(dirs["fingerprint"], f"symbolic_fingerprint_{safe_id}.json")
            universe.memory.export_fingerprint(outpath)

    print("\n✅ Parameter scan completed.")
    print(f"📁 Results: {output_file}")
    print(f"📁 Fingerprints: {dirs['fingerprint']}")
    return dirs

# === Entry Point ===
if __name__ == "__main__":
    grid = {
        "vesica_strength": [0.2, 0.4, 0.6],
        "eta": [0.6, 0.8, 1.0],
    }
    run_scan(grid, steps_per_run=100, run_id=1)
