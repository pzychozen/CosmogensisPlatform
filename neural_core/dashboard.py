import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget, QCheckBox
)
from PyQt5.QtCore import QTimer

from simulation.engine import RecursiveUniverse
from simulation.utils import create_run_dirs


class UniverseDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recursive Universe Dashboard")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- GUI Controls ---
        self.run_button = QPushButton("Start Simulation")
        self.run_button.clicked.connect(self.toggle_simulation)
        self.layout.addWidget(self.run_button)

        self.vesica_toggle = QCheckBox("Enable Vesica Coupling")
        self.vesica_toggle.setChecked(True)
        self.vesica_toggle.stateChanged.connect(self.update_vesica_state)
        self.layout.addWidget(self.vesica_toggle)

        self.status_label = QLabel("Ready.")
        self.layout.addWidget(self.status_label)

        self.data_log = QListWidget()
        self.layout.addWidget(self.data_log)

        # --- Timer for simulation steps ---
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_universe)

        # --- Create folders and universe instance ---
        self.run_id = 1
        self.run_dirs = create_run_dirs(run_id=self.run_id)

        self.universe = RecursiveUniverse(
            use_vesica=True,
            vesica_strength=0.4,
            params={"eta": 0.8},
        )
        self.universe.fingerprint_dir = self.run_dirs["fingerprint"]

        self.running = False

    def toggle_simulation(self):
        self.running = not self.running
        if self.running:
            self.run_button.setText("Pause")
            self.timer.start()
            self.status_label.setText("Running...")
        else:
            self.run_button.setText("Start Simulation")
            self.timer.stop()
            self.status_label.setText("Paused.")

    def update_vesica_state(self):
        self.universe.use_vesica = self.vesica_toggle.isChecked()
        state = "Enabled" if self.universe.use_vesica else "Disabled"
        self.status_label.setText(f"Vesica Coupling {state}.")

    def update_universe(self):
        self.universe.step()
        step_text = f"Step {self.universe.iteration}"
        self.data_log.addItem(step_text)
        self.data_log.scrollToBottom()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = UniverseDashboard()
    dashboard.resize(400, 300)
    dashboard.show()
    sys.exit(app.exec_())
