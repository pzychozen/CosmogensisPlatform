from monitor.meta_recursive_monitor import MetaRecursiveSelfMonitor

if __name__ == "__main__":
    monitor = MetaRecursiveSelfMonitor(refresh_interval=60)  # refresh every 60 seconds
    monitor.start_monitor()
