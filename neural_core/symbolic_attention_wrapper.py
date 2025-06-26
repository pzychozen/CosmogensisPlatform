import torch
import torch.nn as nn
import torch.nn.functional as F

# === Symbolic Attention Wrapper ===
# Injects symbolic memory structure into transformer's attention mechanism

class SymbolicAttentionWrapper(nn.Module):
    def __init__(self, base_model, memory_dim):
        super().__init__()
        self.base_model = base_model
        self.memory_projector = nn.Linear(memory_dim, base_model.config.hidden_size)
        self.memory_gate = nn.Linear(base_model.config.hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask=None, symbolic_memory=None):
        # === Step 1: Embed symbolic memory (optional) ===
        if symbolic_memory is not None:
            projected_memory = self.memory_projector(symbolic_memory)
            # Repeat memory vector for each token in input
            expanded_memory = projected_memory.unsqueeze(1).repeat(1, input_ids.size(1), 1)
        else:
            expanded_memory = torch.zeros(
                input_ids.size(0), input_ids.size(1), self.base_model.config.hidden_size,
                device=input_ids.device
            )

        # === Step 2: Embed input text ===
        embedded_input = self.base_model.get_input_embeddings()(input_ids)

        # === Step 3: Fuse symbolic memory into token embeddings ===
        combined = torch.cat([embedded_input, expanded_memory], dim=-1)
        gate = torch.sigmoid(self.memory_gate(combined))
        fused = embedded_input * (1 - gate) + expanded_memory * gate

        # === Step 4: Run through Transformer encoder ===
        outputs = self.base_model.encoder(
            inputs_embeds=fused,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
