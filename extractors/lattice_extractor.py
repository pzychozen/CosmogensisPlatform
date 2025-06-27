import numpy as np
from scipy.ndimage import maximum_filter, label, center_of_mass
from scipy.spatial import KDTree

# ================================================================
# Symbolic Lattice Extractor (Modular Future-Proof Version)
# ================================================================

def extract_lattice(T, percentile=98, neighborhood=5, connect_radius=None):
    """
    Extract attractor nodes from a tensor field T.

    Parameters:
    - T: 2D numpy array (tensor field)
    - percentile: threshold percentile for activation
    - neighborhood: size of local maximum filter window
    - connect_radius: max distance for node connections (auto-scaled if None)

    Returns:
    - centroids: array of detected node coordinates (N x 2) or empty array
    - pairs: set of connection pairs between nodes
    """
    # Ensure the tensor field T is a 2D numpy array
    if not isinstance(T, np.ndarray) or T.ndim != 2:
        print(f"[ERROR] Invalid tensor shape for extraction: {type(T)}, shape: {getattr(T, 'shape', None)}")
        return np.empty((0, 2)), set()

    grid_size = T.shape[0]  # Assume square grid
    if connect_radius is None:
        connect_radius = int(grid_size * 0.1)  # Auto-scale connect_radius based on grid size

    # --- Detect Attractor Nodes ---
    local_max = maximum_filter(T, size=neighborhood) == T
    try:
        threshold = np.percentile(T, percentile)
    except Exception as e:
        print(f"[ERROR] Percentile computation failed: {e}")
        return np.empty((0, 2)), set()

    attractor_mask = (local_max) & (T > threshold)
    labeled, num_features = label(attractor_mask)
    centroids = np.array(center_of_mass(T, labeled, range(1, num_features + 1)))

    # --- Sanitize Output ---
    if centroids.size == 0 or centroids.shape[0] < 2:
        print(f"[INFO] No attractor clusters found — returning empty lattice.")
        return np.empty((0, 2)), set()

    # --- Ensure centroids is a 2D array (Nx2) ---
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        print(f"[ERROR] Invalid centroids shape: {centroids.shape}. Expecting 2D array with 2 columns.")
        return np.empty((0, 2)), set()

    # --- Build spatial connections ---
    try:
        tree = KDTree(centroids)
        pairs = tree.query_pairs(connect_radius)
    except Exception as e:
        print(f"[ERROR] Failed to compute KDTree connections: {e}")
        return np.empty((0, 2)), set()

    # --- Ensure pairs is a valid set or list ---
    if not isinstance(pairs, (set, list)):
        print(f"[ERROR] Invalid pairs: {type(pairs)}. Expected set or list.")
        return np.empty((0, 2)), set()

    return centroids, pairs

# ================================================================
# END OF MODULE
# ================================================================
