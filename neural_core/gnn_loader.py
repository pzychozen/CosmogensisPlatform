import os
import json
import torch
from torch_geometric.data import Data
import numpy as np

def load_fingerprint_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def fingerprint_to_graphs(fingerprint_path):
    data = load_fingerprint_json(fingerprint_path)
    graphs = []

    for step in data:
        nodes = np.array(step['nodes'])
        edges = np.array(step['edges'])
        iteration = step.get('iteration', 0)

        # Normalize node positions
        x = torch.tensor(nodes, dtype=torch.float)
        x = (x - x.mean(dim=0)) / x.std(dim=0)

        # Convert edge list to edge_index format (2, E)
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, y=torch.tensor([iteration], dtype=torch.float))
        graphs.append(graph)

    return graphs

def load_graphs_from_folder(folder_path):
    all_graphs = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.startswith("symbolic_fingerprint_") and fname.endswith(".json"):
            path = os.path.join(folder_path, fname)
            try:
                graphs = fingerprint_to_graphs(path)
                all_graphs.extend(graphs)
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")
    return all_graphs
