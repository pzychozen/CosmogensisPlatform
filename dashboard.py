import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout

# === Import GUI Tabs ===
from gui.launcher import SimulationLauncher
# Future tabs can be imported here:
# from gui.symbolic_analyzer import SymbolicAnalyzer
# from gui.phase_viewer import PhaseClassifierViewer

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmogensis Platform v2.1")
        self.setGeometry(100, 100, 1000, 700)

        tabs = QTabWidget()
        tabs.addTab(SimulationLauncher(), "Simulation Launcher")

        # Add other GUI tabs here when ready
        # tabs.addTab(SymbolicAnalyzer(), "Symbolic Analyzer")
        # tabs.addTab(PhaseClassifierViewer(), "Phase Classifier")

        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(tabs)
        container.setLayout(layout)
        self.setCentralWidget(container)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec_())
