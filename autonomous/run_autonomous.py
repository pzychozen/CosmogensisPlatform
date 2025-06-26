from autonomous.autonomous_evolution import AutonomousEvolutionEngine

if __name__ == "__main__":
    auto_engine = AutonomousEvolutionEngine()
    auto_engine.evolve(total_generations=10000, steps_per_universe=500, sleep_time=1)
    auto_engine.close()
