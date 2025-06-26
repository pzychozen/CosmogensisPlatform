# gui/main_gui.py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout
from gui.launcher import SimulationLauncher
from gui.dashboard import EvolutionDashboard

class CosmogensisMainGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmogensis Simulation Platform")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        tabs = QTabWidget()

        # Simulation Runner Tab
        self.simulation_launcher = SimulationLauncher()
        tabs.addTab(self.simulation_launcher, "Simulation Runner")

        # Evolution Dashboard Tab
        self.evolution_dashboard = EvolutionDashboard()
        tabs.addTab(self.evolution_dashboard, "Evolution Dashboard")

        layout.addWidget(tabs)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CosmogensisMainGUI()
    window.show()
    sys.exit(app.exec_())
