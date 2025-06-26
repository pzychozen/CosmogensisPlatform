import os
import torch
import random
from torch_geometric.loader import DataLoader
from gnn_loader import load_graphs_from_folder
from symbolic_transformer import SymbolicTransformer

# === Hyperparameters ===
BATCH_SIZE = 1           # sequences, not datapoints
SEQ_LEN = 10             # how many steps per sequence
HIDDEN_DIM = 64
EPOCHS = 30
LR = 1e-3

# === Load and slice data ===
def make_sequence_batches(graphs, seq_len):
    sequences = []
    for i in range(len(graphs) - seq_len):
        seq = graphs[i:i+seq_len]
        motions = []
        for j in range(1, len(seq)):
            prev = seq[j-1].x
            curr = seq[j].x
            if prev.shape == curr.shape:
                diff = (curr - prev).norm(dim=1)
                motions.append(diff.mean().item())
        avg_motion = sum(motions) / len(motions) if motions else 0.0
        sequences.append((seq, torch.tensor([avg_motion])))
    return sequences

# === Main training function ===
def train_symbolic_transformer(fingerprint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    all_graphs = load_graphs_from_folder(fingerprint_dir)
    print(f"Loaded {len(all_graphs)} graphs.")

    data = make_sequence_batches(all_graphs, SEQ_LEN)
    random.shuffle(data)

    model = SymbolicTransformer(hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        model.train()

        for seq, target in data:
            batch_graphs = [g.to(device) for g in seq]
            target = target.to(device)

            optimizer.zero_grad()
            output = model(batch_graphs, None)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        print(f"🌀 Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "symbolic_transformer.pt")
    print("✅ Model saved as symbolic_transformer.pt")
