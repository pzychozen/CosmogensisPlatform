import numpy as np
from scipy.ndimage import maximum_filter, label, center_of_mass
from scipy.spatial import KDTree

# ================================================================
# Symbolic Lattice Extractor (Modular Future-Proof Version)
# ================================================================

def extract_lattice(T, percentile=98, neighborhood=5, connect_radius=10):
    """
    Extract attractor nodes from a tensor field T.

    Parameters:
    - T: 2D numpy array (tensor field)
    - percentile: threshold percentile for activation
    - neighborhood: size of local maximum filter window
    - connect_radius: max distance for node connections

    Returns:
    - centroids: array of detected node coordinates
    - pairs: set of connection pairs between nodes
    """

    # --- Detect Attractor Nodes ---
    local_max = maximum_filter(T, size=neighborhood) == T
    threshold = np.percentile(T, percentile)
    attractor_mask = (local_max) & (T > threshold)

    labeled, num_features = label(attractor_mask)
    centroids = np.array(center_of_mass(T, labeled, range(1, num_features + 1)))

    if centroids.size == 0:
        return None, None

    # --- Build spatial connections ---
    tree = KDTree(centroids)
    pairs = tree.query_pairs(connect_radius)

    return centroids, pairs

# ================================================================
# END OF MODULE
# ================================================================
