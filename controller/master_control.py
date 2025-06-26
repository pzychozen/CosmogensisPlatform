import os
import sys
import time

class MasterControl:
    def __init__(self):
        self.menu()

    def menu(self):
        while True:
            print("\n=== RECURSIVE COSMOGENESIS CONTROL CENTER ===")
            print("1️⃣  Run Single Universe Simulation")
            print("2️⃣  Run Evolutionary Farm Cycle")
            print("3️⃣  Launch Recursive Law Learner")
            print("4️⃣  Open Visual Dashboard")
            print("5️⃣  Display Genome Database Stats")
            print("6️⃣  Exit")

            choice = input("Select option: ")

            if choice == "1":
                self.run_single_universe()
            elif choice == "2":
                self.run_farm_cycle()
            elif choice == "3":
                self.run_law_learner()
            elif choice == "4":
                self.launch_dashboard()
            elif choice == "5":
                self.genome_stats()
            elif choice == "6":
                print("Shutting down...")
                break
            else:
                print("Invalid choice.")

    def run_single_universe(self):
        print("Launching Universe Simulation...")
        os.system("python experiments/main_cosmogenesis.py")

    def run_farm_cycle(self):
        print("Running Evolutionary Farm...")
        os.system("python recursive_farm/farm_manager.py")

    def run_law_learner(self):
        print("Running Recursive Law Learner...")
        os.system("python law_discovery/run_law_learner.py")

    def launch_dashboard(self):
        print("Launching Visual Dashboard...")
        os.system("python dashboard/recursive_dashboard.py")

    def genome_stats(self):
        from database.genome_database import RecursiveGenomeDatabase
        db = RecursiveGenomeDatabase()
        genomes = db.fetch_all_genomes()
        print(f"Total stored genomes: {len(genomes)}")
        db.close()
