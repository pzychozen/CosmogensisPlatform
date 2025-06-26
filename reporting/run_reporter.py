from reporting.self_reflective_reporter import SelfReflectiveKnowledgeReporter

if __name__ == "__main__":
    reporter = SelfReflectiveKnowledgeReporter()
    reporter.save_report()
    reporter.close()
