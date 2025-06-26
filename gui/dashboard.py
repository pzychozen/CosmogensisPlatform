import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget
from PyQt5.QtCore import QTimer
from gui.genome_fetcher import fetch_latest_genomes

class EvolutionDashboard(QWidget):
    def __init__(self, db_path="data_storage/genome_database.db"):
        super().__init__()
        self.db_path = db_path
        self.paused = False
        self.setWindowTitle("Recursive Evolution Dashboard")
        self.setGeometry(100, 100, 600, 400)
        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(3000)

    def setup_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Live Genome Feed")
        self.genome_list = QListWidget()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)

        layout.addWidget(self.label)
        layout.addWidget(self.genome_list)
        layout.addWidget(self.pause_button)
        self.setLayout(layout)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def update_data(self):
        if self.paused:
            return

        genomes = fetch_latest_genomes(self.db_path)
        self.genome_list.clear()
        for iteration, genome in genomes:
            line = f"Iter {iteration}: {''.join(str(g) for g in genome)}"
            self.genome_list.addItem(line)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = EvolutionDashboard()
    dashboard.show()
    sys.exit(app.exec_())
