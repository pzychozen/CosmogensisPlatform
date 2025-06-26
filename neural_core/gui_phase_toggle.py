import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from neural_core.phase_checkpoint_loader import load_phase_classifier
import numpy as np

class PhaseClassifierToggle(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phase Classifier Toggle")

        self.label = QLabel("Model: Not loaded")
        self.toggle_btn = QPushButton("Load Classifier")
        self.toggle_btn.clicked.connect(self.toggle_model)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.toggle_btn)
        self.setLayout(layout)

        self.model = None
        self.model_loaded = False

    def toggle_model(self):
        if not self.model_loaded:
            try:
                data = np.load("symbolic_embeddings.npz")
                input_dim = data["X"].shape[1]
                self.model = load_phase_classifier(input_dim, auto_retrain=True)
                self.model_loaded = True
                self.label.setText("Model: Loaded ✅")
                self.toggle_btn.setText("Unload Classifier")
            except Exception as e:
                self.label.setText(f"Error: {e}")
        else:
            self.model = None
            self.model_loaded = False
            self.label.setText("Model: Unloaded ❌")
            self.toggle_btn.setText("Load Classifier")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhaseClassifierToggle()
    window.show()
    sys.exit(app.exec_())
