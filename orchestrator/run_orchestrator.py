from orchestrator.experiment_orchestrator import RecursiveExperimentOrchestrator

if __name__ == "__main__":
    orchestrator = RecursiveExperimentOrchestrator()
    orchestrator.start_long_term_experiment(days=30)
