from law_discovery.recursive_law_learner import RecursiveLawLearner

if __name__ == "__main__":
    learner = RecursiveLawLearner()
    common_patterns = learner.discover_laws(chunk_size=4)
    learner.visualize_law_distribution(common_patterns)
    learner.close()
