# ===========================
# File: core_engine/recursive_universe.py
# ===========================

import numpy as np
from scipy.ndimage import maximum_filter, label, center_of_mass
from scipy.spatial import KDTree
from neural_core.gnn_loader import fingerprint_to_graphs
from neural_core.symbolic_phase_classifier import PhaseClassifier

from extractors.lattice_extractor import extract_lattice
from extractors.memory_lattice import MemoryLattice


class RecursiveUniverse:
    def __init__(self, grid_size=128, params=None):
        self.grid_size = grid_size
        self.T = self.initialize_tensor_field()
        self.Structured_Time = np.zeros((grid_size, grid_size))
        self.theta_phase = 0.0
        self.iteration = 0
        self.memory = MemoryLattice()

        self.params = params or {
            'alpha': 1.0, 'beta': 0.2, 'gamma': 0.05, 'delta': 0.01,
            'eta': 0.8, 'epsilon': 0.002, 'lambda_amp': 0.3,
            'alpha_phase': 1.0, 'time_coupling': 0.005,
            'time_feedback': 0.01, 'fusion_strength': 0.3,
            'tunneling_rate': 0.002, 'Q_max': 100
        }

    def initialize_tensor_field(self):
        return 0.01 * np.random.randn(self.grid_size, self.grid_size)

    def initialize_dynamic_vesica(self):
        x = np.linspace(-1, 1, self.grid_size)
        y = np.linspace(-1, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        shift = 0.5 * np.cos(self.Structured_Time.mean())
        r1 = np.sqrt((X - shift)**2 + Y**2)
        r2 = np.sqrt((X + shift)**2 + Y**2)
        return np.exp(-10 * r1**2) + np.exp(-10 * r2**2)

    def gradient_complex(self, field):
        grad_real_x = np.gradient(field.real, axis=0)
        grad_real_y = np.gradient(field.real, axis=1)
        grad_imag_x = np.gradient(field.imag, axis=0)
        grad_imag_y = np.gradient(field.imag, axis=1)
        return (grad_real_x + 1j * grad_imag_x, grad_real_y + 1j * grad_imag_y)

    def divergence_complex(self, grad_x, grad_y):
        div_x = np.gradient(grad_x.real, axis=0) + 1j * np.gradient(grad_x.imag, axis=0)
        div_y = np.gradient(grad_y.real, axis=1) + 1j * np.gradient(grad_y.imag, axis=1)
        return div_x + div_y

    def laplacian(self, field):
        fx = np.gradient(field, axis=0)
        fy = np.gradient(field, axis=1)
        return np.gradient(fx, axis=0) + np.gradient(fy, axis=1)

    def E(self, T):
        return 0.01 * T

    def compute_tensor_feedback(self, T):
        return -0.05 * T**3

    def step(self):
        p = self.params

        phi = self.initialize_dynamic_vesica()
        Phi = np.exp(1j * self.theta_phase) * self.T

        grad_x, grad_y = self.gradient_complex(Phi)
        div_Phi = self.divergence_complex(grad_x, grad_y)
        kappa_Phi = div_Phi - self.laplacian(Phi.real)

        Q_base = np.floor(p['alpha'] * kappa_Phi.real + p['beta'] * self.E(self.T))
        Q_capped = np.clip(Q_base, -p['Q_max'], p['Q_max'])

        Lambda = np.sin(p['gamma'] * Q_capped) * np.exp(1j * self.theta_phase) * np.exp(-p['delta'] * self.T)
        kappa_RDPTF = kappa_Phi.real + p['eta'] * np.real(Lambda)
        Q_RDPTF = np.floor(p['alpha'] * kappa_RDPTF + p['beta'] * self.E(self.T))

        leakage = np.tanh(Q_RDPTF - np.mean(Q_RDPTF))
        noise = p['epsilon'] * np.random.randn(self.grid_size, self.grid_size)
        leakage_field = leakage + noise

        V_theta = np.sin(3 * self.Structured_Time + phi) + np.sin(5 * self.Structured_Time + phi)
        tensor_feedback = self.compute_tensor_feedback(self.T)
        breathing_amp = (-p['alpha_phase'] * V_theta + tensor_feedback) * (1 + p['lambda_amp'] * np.exp(-np.abs(self.T)))
        breathing_amp += 0.05 * leakage_field

        fusion_field = p['fusion_strength'] * np.tanh(self.T - self.Structured_Time)
        tunneling = p['tunneling_rate'] * np.random.randn(self.grid_size, self.grid_size)

        T_new = np.abs(Q_RDPTF) / (1 + np.abs(Q_RDPTF)) + breathing_amp + fusion_field + tunneling
        self.T = T_new

        time_energy = np.abs(Q_RDPTF)
        self.Structured_Time += p['time_coupling'] * (time_energy - self.Structured_Time) + p['time_feedback'] * leakage

        self.theta_phase += 0.1 * np.mean(np.abs(Q_RDPTF))

        # NEW: inline symbolic phase prediction (in step)
        if self.iteration % 50 == 0:
            centroids, pairs = extract_lattice(self.T)
            self.memory.store(self.iteration, centroids, pairs)

            # 🔮 Load GNN+Transformer → embed
            from neural_core.gnn_loader import fingerprint_to_graphs
            from neural_core.symbolic_transformer import SymbolicTransformer
            from neural_core.symbolic_phase_classifier import PhaseClassifier

            snapshot = [{"iteration": self.iteration, "nodes": centroids.tolist(), "edges": list(map(list, pairs))}]
            graphs = fingerprint_to_graphs(snapshot)

            # Encode graph → latent vector
            transformer = SymbolicTransformer()
            transformer.load_state_dict(torch.load("symbolic_transformer.pt"))
            transformer.eval()

            latent = transformer.encoder(graphs[0].x, graphs[0].edge_index, torch.zeros(graphs[0].x.shape[0], dtype=torch.long).to(graphs[0].x.device))

            # Classify phase
            classifier = PhaseClassifier()
            classifier.load_state_dict(torch.load("symbolic_phase_classifier.pt"))
            classifier.eval()

            phase_idx = torch.argmax(classifier(latent.unsqueeze(0))).item()
            phase_names = ["stable", "recursive", "chaotic"]
            print(f"[{self.iteration}] Predicted Phase: {phase_names[phase_idx]}")

            self.iteration += 1