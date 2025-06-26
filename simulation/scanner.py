# === PATCHED SCANNER ===

import itertools
import csv
import os
import numpy as np
from datetime import datetime
from simulation.engine import RecursiveUniverse
from simulation.utils import create_run_dirs

# === CONFIG ===
default_params = {
    "vesica_strength": [0.2, 0.4],
    "eta": [0.6, 0.8],
    "steps_per_run": 100,
    "snapshot_interval": 10
}

# === HEADER ===
result_fields = ["vesica_strength", "eta", "iteration", "node_count", "avg_motion"]

# === MAIN ENTRY ===
def run_scan(grid, steps_per_run=100, run_id=1):
    dirs = create_run_dirs(run_id)
    run_dir = dirs["analyzis"]
    os.makedirs("data/fingerprint/default", exist_ok=True)

    results = []
    counter = 0

    for params in grid:
        print(f"\n\u2699\ufe0f Running: {params}")
        universe = RecursiveUniverse(
        grid_size=50,
        use_vesica=True,
        vesica_strength=params["vesica_strength"],
        params={"eta": params["eta"]}
    )


        # Run steps and record snapshots manually
        for i in range(steps_per_run):
            universe.step()
            counter += 1
            if i % default_params["snapshot_interval"] == 0:
                universe.lattice_manager.process(universe.T, universe.iteration)

        # Save symbolic fingerprint after run
        symbolic_path = f"data/fingerprint/default/symbolic_fingerprint_{universe.iteration}.json"
        universe.memory.export_fingerprint(symbolic_path)

        motions = universe.memory.compute_motion_vectors()
        if motions:
            motion_values = [m["magnitudes"].mean() for m in motions if m["magnitudes"].size > 0]
            avg_motion = np.mean(motion_values) if motion_values else 0.0
        else:
            avg_motion = 0.0

        result_row = [params["vesica_strength"], params["eta"], universe.iteration, universe.lattice_manager.node_count(), avg_motion]
        results.append(result_row)

        print(f"\u2705 Exported {universe.memory.step_count()} symbolic fingerprint steps to {symbolic_path}")
        print(f"[motion] Computed motion vectors for {len(motions)} steps.")

    # === SAVE RESULTS ===
    output_csv = os.path.join(run_dir, "latest_run_results.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result_fields)
        writer.writerows(results)

    print("\n[✓] Parameter scan completed.")
    print(f"[results] File saved to: {output_csv}")
    print(f"[fingerprints] Location: {run_dir.replace('analyzis', 'fingerprint')}")


# === GRID BUILDER ===
def build_param_grid(param_ranges):
    keys = param_ranges.keys()
    values = list(itertools.product(*param_ranges.values()))
    return [dict(zip(keys, v)) for v in values]


if __name__ == "__main__":
    param_grid = build_param_grid({"vesica_strength": [0.2, 0.4], "eta": [0.6, 0.8]})
    run_scan(param_grid, steps_per_run=100, run_id=1)