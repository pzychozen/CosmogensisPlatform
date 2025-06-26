import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

LABELS = {
    "stable": 0,
    "recursive": 1,
    "chaotic": 2
}

def auto_label_motion(motion_val):
    if motion_val < 1.0:
        return LABELS["stable"]
    elif motion_val < 2.0:
        return LABELS["recursive"]
    else:
        return LABELS["chaotic"]

class PhaseClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_phase_classifier(embeddings, motion_targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor([auto_label_motion(m) for m in motion_targets], dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PhaseClassifier(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "symbolic_phase_classifier.pt")
    print("✅ Saved classifier to symbolic_phase_classifier.pt")
    return model
