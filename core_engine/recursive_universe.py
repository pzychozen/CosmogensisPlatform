import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import KDTree
from neural_core.gnn_loader import fingerprint_to_graphs
from neural_core.symbolic_phase_classifier import PhaseClassifier
from extractors.lattice_extractor import extract_lattice
from extractors.memory_lattice import MemoryLattice

class UnifiedRecursiveUniverse:
    def __init__(self, grid_size=128, vesica_mode='fixed', params=None, fingerprint_dir="data/fingerprint/default"):
        self.grid_size = grid_size
        self.vesica_mode = vesica_mode  # 'fixed', 'dynamic', or 'both'
        self.T = self.initialize_tensor_field()
        self.Structured_Time = np.zeros((grid_size, grid_size))
        self.theta_phase = 0.0
        self.iteration = 0
        self.memory = MemoryLattice()
        self.fingerprint_dir = fingerprint_dir
        self.lattice_manager = None  # Will handle lattice management if needed

        # Use default params if None are provided
        self.default_params = {
            'alpha': 1.0, 'beta': 0.2, 'gamma': 0.05, 'delta': 0.01,
            'eta': 0.8, 'epsilon': 0.002, 'lambda_amp': 0.3,
            'alpha_phase': 1.0, 'time_coupling': 0.005,
            'time_feedback': 0.01, 'fusion_strength': 0.3,
            'tunneling_rate': 0.002, 'Q_max': 100,
            'vesica_strength': 0.4  # Add this to the default params
        }

        # Initialize the params either from the argument or use defaults
        self.params = {**self.default_params, **(params or {})}  # Merge default params with provided ones
        print(f"[DEBUG] Params after merge: {self.params}")  # Debugging line to check merged params

        # Ensure that we have all required parameters
        required_params = ['alpha', 'beta', 'gamma', 'delta', 'eta', 'epsilon', 'lambda_amp', 'alpha_phase', 'time_coupling', 'time_feedback', 'fusion_strength', 'tunneling_rate', 'Q_max', 'vesica_strength']
        for param in required_params:
            if param not in self.params:
                print(f"[ERROR] Missing parameter '{param}' in params!")
                self.params[param] = self.default_params[param]  # Fallback to default param if missing

        print(f"[DEBUG] Final params in step: {self.params}")

        required_params = ['alpha', 'beta', 'gamma', 'delta', 'eta', 'epsilon', 'lambda_amp', 'alpha_phase', 'time_coupling', 'time_feedback', 'fusion_strength', 'tunneling_rate', 'Q_max', 'vesica_strength']
        for param in required_params:
            if param not in self.params:
                print(f"[Warning] Missing parameter: {param}, using default value.")
                self.params[param] = self.params.get(param, 1.0)

        print(f"[DEBUG] Params after merge: {self.params}")  # Debugging line to check merged params

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

    def compute_motion_vectors(self):
        motions = []
        for i in range(1, len(self.centroids)):
            # Ensure centroids are in 2D form
            current_centroid = np.array(self.centroids[i])
            previous_centroid = np.array(self.centroids[i-1])

            # If the centroids are not 2D, we can't compute the motion, skip this iteration
            if current_centroid.ndim != 2 or previous_centroid.ndim != 2:
                print(f"[ERROR] Invalid centroids at iteration {i}, skipping motion vector calculation.")
                continue
        
            # Compute displacement using Euclidean distance
            displacement = np.linalg.norm(current_centroid - previous_centroid, axis=1)
            motions.append(displacement)
    
        return motions


    def update_centroids(self):
        centroids, pairs = extract_lattice(self.T)

        # Debugging: Check types and shapes of centroids and pairs
        print(f"[DEBUG] Iteration {self.iteration}: centroids type: {type(centroids)}, shape: {getattr(centroids, 'shape', None)}")
        print(f"[DEBUG] Iteration {self.iteration}: pairs type: {type(pairs)}, length: {len(pairs) if isinstance(pairs, (list, set)) else 'N/A'}")

        # Ensure valid centroids and pairs before storing
        if not isinstance(centroids, np.ndarray) or centroids.ndim != 2 or centroids.shape[1] != 2:
            print(f"[ERROR] Invalid centroids at iteration {self.iteration}: {centroids}")
            return

        if not isinstance(pairs, (list, set)):
            print(f"[ERROR] Invalid pairs at iteration {self.iteration}: {pairs}")
            return

        self.centroids = centroids  # Save centroids for motion vector calculation
        self.memory.store(centroids, pairs, self.iteration)  # Store centroids and pairs in memory


    def step(self):
        print(f"[DEBUG] Current params in step: {self.params}")  # Add this line to check the params
        p = self.params

        # Vesica strength handling based on mode
        if self.vesica_mode == 'dynamic':
            vesica_strength = 0.4 + 0.2 * np.sin(self.iteration / 50)
        elif self.vesica_mode == 'fixed':
            vesica_strength = 0.4  # Fixed value for Vesica strength
        else:
            vesica_strength = 0.4 + 0.2 * np.sin(self.iteration / 50)  # Hybrid mode

        phi = vesica_strength * self.initialize_dynamic_vesica() if self.vesica_mode != 'fixed' else 0.0
        Phi = np.exp(1j * self.theta_phase) * self.T

        grad_x, grad_y = self.gradient_complex(Phi)
        div_Phi = self.divergence_complex(grad_x, grad_y)
        kappa_Phi = div_Phi - self.laplacian(Phi.real)

        # Check if 'alpha' and 'beta' exist in params
        if 'alpha' not in p or 'beta' not in p:
            print(f"[ERROR] Missing parameter 'alpha' or 'beta' in params: {p}")
            return  # Or handle the error accordingly

        Q_base = np.floor(p['alpha'] * kappa_Phi.real + p['beta'] * self.E(self.T))
        Q_capped = np.clip(Q_base, -p['Q_max'], p['Q_max'])

        Lambda = np.sin(p['gamma'] * Q_capped) * np.exp(1j * self.theta_phase) * np.exp(-p['delta'] * self.T)
        kappa_RDPTF = kappa_Phi.real + p['eta'] * np.real(Lambda)
        Q_RDPTF = np.floor(p['alpha'] * kappa_RDPTF + p['beta'] * self.E(self.T))

        leakage = np.tanh(Q_RDPTF - np.mean(Q_RDPTF))
        leakage_field = leakage + p['epsilon'] * np.random.randn(self.grid_size, self.grid_size)

        # Breathing amplifier
        breathing_amp = (-p['alpha_phase'] * np.sin(3 * self.Structured_Time + phi) + self.compute_tensor_feedback(self.T)) * (1 + p['lambda_amp'] * np.exp(-np.abs(self.T))) + 0.05 * leakage_field

        T_new = np.abs(Q_RDPTF) / (1 + np.abs(Q_RDPTF)) + breathing_amp

        self.Structured_Time += p['time_coupling'] * (np.abs(Q_RDPTF) - self.Structured_Time) + p['time_feedback'] * leakage
        self.theta_phase += 0.1 * np.mean(np.abs(Q_RDPTF))

        self.iteration += 1

        # 1. Update centroids (extract lattice information)
        self.update_centroids()

        # 2. Compute and log motion vectors
        motions = self.compute_motion_vectors()
        if motions:
            avg_motion = np.mean([np.mean(m) for m in motions if m.size > 0])
            print(f"Iteration {self.iteration} — avg motion magnitude: {avg_motion:.5f}")
        else:
            avg_motion = 0.0

        return T_new


    def symbolic_phase_prediction(self):
        # Inline symbolic phase prediction (similar to `recursive_universe.py`)
        if self.iteration % 50 == 0:
            centroids, pairs = extract_lattice(self.T)
            self.memory.store(self.iteration, centroids, pairs)

            # 🔮 Load GNN+Transformer → embed
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