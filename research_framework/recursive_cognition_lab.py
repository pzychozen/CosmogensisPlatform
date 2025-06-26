import os
import time
import datetime
import subprocess
from pathlib import Path
from neural_core.gnn_loader import fingerprint_to_graphs
from neural_core.symbolic_phase_classifier import PhaseClassifier


class RecursiveCognitionLab:
    def __init__(self):
        self.log("Recursive Cognition Research Framework initialized.")
        self.daily_cycles = 0
        self.project_root = Path(__file__).resolve().parent.parent  # Assumes structure consistency

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def run_script(self, script_path):
        try:
            subprocess.run(["python", script_path], check=True)
        except subprocess.CalledProcessError as e:
            self.log(f"Error running {script_path}: {e}")

    def run_daily_research_cycle(self):
        self.log("Running full autonomous recursive cognition cycle...")

        # 1️⃣ Evolve new recursive genomes
        self.run_script(self.project_root / "autonomous" / "run_autonomous.py")

        # 2️⃣ Compress symbolic laws
        self.run_script(self.project_root / "law_discovery" / "run_law_learner.py")

        # 3️⃣ Run meta-law compression
        self.run_script(self.project_root / "meta_compression" / "run_meta_compression.py")

        # 4️⃣ Generate self-reflective report
        self.run_script(self.project_root / "reporting" / "run_reporter.py")

        # 5️⃣ Backup symbolic memory
        backup_dir = self.project_root / "data_storage" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        src = self.project_root / "data_storage" / "symbolic_memory.db"
        dst = backup_dir / f"memory_backup_{timestamp}.db"

        if src.exists():
            dst.write_bytes(src.read_bytes())
            self.log(f"Symbolic memory backed up to {dst}")
        else:
            self.log("Warning: symbolic_memory.db not found. Backup skipped.")

        self.daily_cycles += 1
        self.log(f"Daily cognition cycle {self.daily_cycles} complete.")

    def run_long_term_experiment(self, total_days=365):
        self.log(f"Launching {total_days}-day long-term recursion experiment...")
        for day in range(total_days):
            self.log(f"=== DAY {day + 1} ===")
            self.run_daily_research_cycle()
            time.sleep(1)  # For full 24-hour delay: use time.sleep(86400)
        self.log("Long-term recursive cognition experiment complete.")

