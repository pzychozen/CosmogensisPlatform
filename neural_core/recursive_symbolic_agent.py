import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from neural_core.symbolic_transformer import SymbolicTransformer
from neural_core.symbolic_phase_classifier import PhaseClassifier, auto_label_motion
from neural_core.symbolic_attention_wrapper import SymbolicAttentionWrapper
from neural_core.gnn_loader import load_symbolic_graph


# === Load Components ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
base_model = AutoModel.from_pretrained("bert-base-uncased")
transformer = SymbolicTransformer(input_dim=64, embed_dim=128)
classifier = PhaseClassifier(input_dim=128)

# Wrap base model to include symbolic memory
model = SymbolicAttentionWrapper(base_model, memory_dim=128)

# === Recursive Memory Agent ===
class RecursiveSymbolicAgent:
    def __init__(self):
        self.symbolic_memory = torch.zeros(1, 128)  # start with null memory
        self.dialogue_history = []

    def observe(self, text):
        self.dialogue_history.append(text)
        inputs = tokenizer(text, return_tensors="pt")
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], symbolic_memory=self.symbolic_memory)
        hidden = output[:, 0, :]  # CLS token embedding

        # Update symbolic memory via symbolic transformer
        # Simulate generating new fingerprint from internal state
        dummy_graph = load_symbolic_graph("data/fingerprint/default/symbolic_fingerprint_50.json")
        with torch.no_grad():
            embedding = transformer(dummy_graph.x, dummy_graph.edge_index)
            new_mem = torch.mean(embedding, dim=0, keepdim=True)

        # Classify and evolve memory based on phase
        phase_logits = classifier(new_mem)
        phase = torch.argmax(phase_logits, dim=1).item()
        phase_tag = ["stable", "recursive", "chaotic"][phase]
        print(f"🌀 Current Phase: {phase_tag}")

        # Recursive memory update
        self.symbolic_memory = 0.7 * self.symbolic_memory + 0.3 * new_mem

        return phase_tag, new_mem

# === Example Loop ===
if __name__ == "__main__":
    agent = RecursiveSymbolicAgent()
    while True:
        text = input("You: ")
        phase, mem = agent.observe(text)
        print(f"[Agent phase: {phase}] updated memory norm: {mem.norm().item():.3f}")
