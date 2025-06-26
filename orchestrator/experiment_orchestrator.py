import time
import datetime

from autonomous.autonomous_evolution import AutonomousEvolutionEngine
from law_discovery.recursive_law_learner import RecursiveLawLearner
from analyzers.entropy_dynamics_analyzer import EntropyDynamicsAnalyzer
from database.genome_database import RecursiveGenomeDatabase
from adaptive.recursive_optimizer import RecursiveAdaptiveOptimizer

class RecursiveExperimentOrchestrator:
    def __init__(self):
        self.auto_engine = AutonomousEvolutionEngine()
        self.db = RecursiveGenomeDatabase()
        self.log("Orchestrator Initialized.")

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def run_daily_batch(self, generations_per_day=100, steps_per_universe=500):
        self.log("Running daily autonomous evolution batch...")
        self.auto_engine.evolve(
            total_generations=generations_per_day,
            steps_per_universe=steps_per_universe,
            sleep_time=0
        )

    def run_law_discovery(self):
        self.log("Running symbolic law compression analysis...")
        law_learner = RecursiveLawLearner()
        common_patterns = law_learner.discover_laws(chunk_size=4)
        # Save law candidates for external analysis
        with open(f"data_storage/law_candidates/law_candidates_{self.timestamp()}.txt", 'w') as f:
            for pattern, count in common_patterns:
                f.write(f"{pattern} — {count}\n")
        law_learner.close()
        self.log("Law extraction complete.")

    def run_entropy_analysis(self):
        self.log("Running entropy dynamics analysis...")
        entropy_analyzer = EntropyDynamicsAnalyzer(self.auto_engine.memory)
        iterations, entropies = entropy_analyzer.analyze_full_memory()
        # Save entropy logs
        with open(f"data_storage/entropy_logs/entropy_{self.timestamp()}.txt", 'w') as f:
            for iter, ent in zip(iterations, entropies):
                f.write(f"{iter},{ent}\n")
        self.log("Entropy analysis complete.")

    def backup_database(self):
        self.log("Backing up genome database...")
        import shutil
        shutil.copy(
            "data_storage/genome_database.db",
            f"data_storage/backups/genome_backup_{self.timestamp()}.db"
        )
        self.log("Backup complete.")

    def timestamp(self):
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def full_daily_cycle(self):
        self.log("Starting full daily experiment cycle...")
        self.run_daily_batch(generations_per_day=100)
        self.run_law_discovery()
        self.backup_database()
        self.log("Daily cycle complete.")

    def start_long_term_experiment(self, days=30):
        self.log(f"Starting long-term experiment: {days} days.")
        for day in range(days):
            self.log(f"=== DAY {day + 1} ===")
            self.full_daily_cycle()
            time.sleep(1)  # In production: replace with 86400 for true daily cycle
        self.auto_engine.close()
        self.db.close()
        self.log("Long-term experiment complete.")

    def run_adaptive_optimization(self):
        optimizer = RecursiveAdaptiveOptimizer()
        adjustments = optimizer.propose_parameter_adjustments()
        # In future: apply adjustments into evolutionary farm
        optimizer.close()
