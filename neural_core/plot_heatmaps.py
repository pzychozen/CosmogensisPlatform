import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def plot_heatmaps(csv_path="data/analyzis/latest_run_results.csv", run_id=1):
    # Load CSV data
    if not os.path.exists(csv_path):
        print(f"[SKIP] No CSV file at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[SKIP] CSV is empty, skipping heatmap generation.")
        return

    # Round and check essential columns
    if not all(col in df.columns for col in ["vesica_strength", "eta", "avg_motion", "node_count"]):
        print("[SKIP] Required columns missing in CSV.")
        return

    df['vesica_strength'] = df['vesica_strength'].round(3)
    df['eta'] = df['eta'].round(3)

    # Pivot tables
    motion_matrix = df.pivot(index="vesica_strength", columns="eta", values="avg_motion")
    node_matrix = df.pivot(index="vesica_strength", columns="eta", values="node_count")

    if motion_matrix.isnull().all().all() or node_matrix.isnull().all().all():
        print("[SKIP] No valid data in pivot tables. Skipping heatmap.")
        return

    # Create folders only when needed
    timestamp = datetime.now().strftime("%d.%m_%H.%M")
    base = f"run_{run_id}_{timestamp}"
    out_dir = os.path.join("data", "heatmap", base)
    os.makedirs(out_dir, exist_ok=True)

    # Average Motion Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(motion_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'Avg Motion'})
    plt.title("Average Motion Heatmap")
    plt.xlabel("Eta")
    plt.ylabel("Vesica Strength")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_avg_motion.png"))
    plt.close()

    # Node Count Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(node_matrix, annot=True, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Node Count'})
    plt.title("Node Count Heatmap")
    plt.xlabel("Eta")
    plt.ylabel("Vesica Strength")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_node_count.png"))
    plt.close()

    print(f"✅ Heatmaps saved to: {out_dir}")

if __name__ == "__main__":
    plot_heatmaps()
