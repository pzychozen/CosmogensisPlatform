from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal
import sqlite3
import json

from simulation.scanner import run_scan, build_param_grid
from gui.genome_fetcher import fetch_latest_genomes

# Function to fetch the latest genomes from the database
# Correct database path
db_path = 'data_storage/genome_database.db'  # Ensure this is your actual path

def fetch_latest_genomes(db_path, limit=10):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT iteration, genome FROM genomes ORDER BY id DESC LIMIT ?"
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [(it, json.loads(genome)) for it, genome in rows]

# Now use the fetch_latest_genomes function correctly in your main simulation
def run_simulation():
    self.console.append("Launching scan...\n")
    param_grid = build_param_grid({
        "vesica_strength": [0.2, 0.4],
        "eta": [0.6, 0.8]
    })
    run_scan(param_grid, steps_per_run=100, run_id=1)
    self.console.append("✅ Scan complete.\n")
    
    # Fetch the latest genomes after running the simulation
    latest_genomes = fetch_latest_genomes(db_path, limit=10)
    for it, genome in latest_genomes:
        self.console.append(f"Iteration {it} Genome: {genome}")

class SimulationThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def run(self):
        param_grid = build_param_grid({
            "vesica_strength": [0.2, 0.4],
            "eta": [0.6, 0.8]
        })
        
        # Loop to call symbolic genome evolution, run the simulation, and update
        for params in param_grid:
            self.update_signal.emit(f"Running simulation with {params}...")
            run_scan(param_grid, steps_per_run=100, run_id=1)

            # Fetch the latest genome after each step and display it in the console
            latest_genomes = fetch_latest_genomes(db_path, limit=10)
            for it, genome in latest_genomes:
                self.update_signal.emit(f"Iteration {it} Genome: {genome}")

            self.update_signal.emit(f"Simulation complete for {params}")

        self.finished_signal.emit()

class SimulationLauncher(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.launch_simulation)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def launch_simulation(self):
        self.console.append("Launching scan...\n")
        self.run_button.setEnabled(False)

        self.thread = SimulationThread()
        self.thread.update_signal.connect(self.console.append)  # Stream output to GUI
        self.thread.finished_signal.connect(self.on_scan_complete)
        self.thread.start()

    def on_scan_complete(self):
        self.console.append("\n✅ Scan complete.\n")
        self.run_button.setEnabled(True)
