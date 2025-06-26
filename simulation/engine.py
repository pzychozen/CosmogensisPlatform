import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import KDTree

from .utils import create_run_dirs
from neural_core.lattice_extractor import extract_lattice
from .memory_lattice import MemoryLattice
from .lattice_manager import LatticeManager


class RecursiveUniverse:
    def __init__(self, grid_size=128, use_vesica=True, vesica_strength=1.0, params=None, fingerprint_dir=None):
        self.grid_size = grid_size
        self.T = self.initialize_tensor_field()
        self.Structured_Time = np.zeros((grid_size, grid_size))
        self.theta_phase = 0.0
        self.iteration = 0
        self.use_vesica = use_vesica
        self.vesica_strength = vesica_strength
        self.memory = MemoryLattice()
        self.lattice_manager = LatticeManager(extract_lattice, self.memory)
        self.vesica_scheduler = lambda i: 0.4 + 0.2 * np.sin(i / 50)

        self.fingerprint_dir = fingerprint_dir or "data/fingerprint/default"
        os.makedirs(self.fingerprint_dir, exist_ok=True)

        default_params = {
            'alpha': 1.0, 'beta': 0.2, 'gamma': 0.05, 'delta': 0.01,
            'eta': 0.8, 'epsilon': 0.002, 'lambda_amp': 0.3,
            'alpha_phase': 1.0, 'time_coupling': 0.005,
            'time_feedback': 0.01, 'fusion_strength': 0.3,
            'tunneling_rate': 0.002, 'Q_max': 100
        }
        if params:
            print(f"Overriding parameters: {params}")
            default_params.update(params)
        self.params = default_params

    def initialize_tensor_field(self):
        return 0.01 * np.random.randn(self.grid_size, self.grid_size)

    def initialize_dynamic_vesica(self):
        x = np.linspace(-1, 1, self.grid_size)
        y = np.linspace(-1, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        shift = 0.5 * np.cos(self.Structured_Time.mean())
        r1 = np.sqrt((X - shift) ** 2 + Y ** 2)
        r2 = np.sqrt((X + shift) ** 2 + Y ** 2)
        return np.exp(-10 * r1 ** 2) + np.exp(-10 * r2 ** 2)

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
        return -0.05 * T ** 3

    def create_run_dirs(run_id=1):
        base_root = "data"
        os.makedirs(base_root, exist_ok=True)

        timestamp = datetime.now().strftime("%d.%m_%H.%M")
        base = f"run_{run_id}_{timestamp}"
        dirs = {
            "fingerprint": os.path.join(base_root, "fingerprint", base),
            "analyzis": os.path.join(base_root, "analyzis", base)
        }
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        return dirs


    def step(self):
        p = self.params
        self.vesica_strength = self.vesica_scheduler(self.iteration)
        phi = self.vesica_strength * self.initialize_dynamic_vesica() if self.use_vesica else 0.0

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
        self.iteration += 1

        if self.iteration % 50 == 0:
            self.lattice_manager.process(self.T, self.iteration)
            export_path = os.path.join(self.fingerprint_dir, f"symbolic_fingerprint_{self.iteration}.json")
            self.memory.export_fingerprint(export_path)


    def finalize_run(self):
        motions = self.memory.compute_motion_vectors()
        if motions:
            avg_motion = np.mean([
                m['average_magnitude'] for m in motions
                if m.get('average_magnitude') is not None
            ])
            print(f"🔬 Overall Average Motion: {avg_motion:.5f}")
        else:
            avg_motion = 0.0
            print("⚠️ No motion vectors computed.")

        return {
            "node_count": len(self.memory.last_centroids) if self.memory.last_centroids is not None else 0,
            "avg_motion": avg_motion
        }

    def save_lattice_image(self, filename="lattice.png"):
        centroids, pairs = extract_lattice(self.T)
        plt.imshow(self.T, cmap='gray')
        if isinstance(centroids, np.ndarray) and centroids.ndim == 2 and centroids.shape[0] > 0:
            plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=30)
            if pairs and isinstance(pairs, (set, list)):
                for i, j in pairs:
                    try:
                        x0, y0 = centroids[i]
                        x1, y1 = centroids[j]
                        plt.plot([y0, y1], [x0, x1], 'b-', linewidth=1)
                    except IndexError:
                        continue
        plt.title("Lattice Extraction")
        plt.savefig(filename)
        plt.close()


def create_run_dirs(run_id=1):
    timestamp = datetime.now().strftime("%d.%m_%H.%M")
    base = f"run_{run_id}_{timestamp}"
    dirs = {
        "fingerprint": os.path.join("data", "fingerprint", base),
        "analyzis": os.path.join("data", "analyzis", base)
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


if __name__ == "__main__":
    dirs = create_run_dirs(run_id=1)
    universe = RecursiveUniverse(
        use_vesica=True,
        fingerprint_dir=dirs["fingerprint"]
    )

    for _ in range(300):
        universe.step()

    universe.save_lattice_image(os.path.join(dirs["fingerprint"], "lattice.png"))
    universe.memory.export_fingerprint(os.path.join(dirs["fingerprint"], "symbolic_fingerprint.json"))

# Optional: Heatmap Plot (if CSV is generated externally)
try:
    from plot_heatmaps import plot_heatmaps
    plot_heatmaps(run_id=1)
except Exception as e:
    print(f"[WARNING] Heatmap generation failed: {e}")

