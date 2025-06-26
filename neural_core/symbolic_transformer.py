import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class SymbolicGraphEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        out = self.pool(x, batch)  # → [batch_size, hidden_dim]
        return out

class SymbolicTransformer(nn.Module):
    def __init__(self, hidden_dim=64, seq_len=10, num_heads=4):
        super().__init__()
        self.encoder = SymbolicGraphEncoder(hidden_dim=hidden_dim)

        self.pos_emb = nn.Parameter(torch.randn(seq_len, hidden_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=2
        )
        self.fc_out = nn.Linear(hidden_dim, 1)  # Output: motion prediction (or stability)

    def forward(self, batch_graphs, batch_vector):
        # Encode graphs into vectors
        reps = []
        for data in batch_graphs:
            rep = self.encoder(data.x, data.edge_index, data.batch)
            reps.append(rep)
        x = torch.stack(reps, dim=0)  # [seq_len, batch_size, hidden_dim]

        # Add positional embedding
        x = x + self.pos_emb[:x.size(0)]

        # Transformer expects [seq_len, batch, dim]
        out = self.transformer(x)
        final = out[-1]  # Use last token
        pred = self.fc_out(final)
        return pred
