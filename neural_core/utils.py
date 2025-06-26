import os
from datetime import datetime

def create_run_dirs(run_id=1):
    timestamp = datetime.now().strftime("%d.%m_%H.%M")
    base = f"run_{run_id}_{timestamp}"
    dirs = {
        "fingerprint": os.path.join("data", "fingerprint", base),
        "heatmap": os.path.join("data", "heatmap", base),
        "analyzis": os.path.join("data", "analyzis", base),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs
