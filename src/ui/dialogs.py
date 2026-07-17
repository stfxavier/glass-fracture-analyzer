"""Dialog windows for the application."""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton
from PyQt6.QtCore import Qt


class SettingsDialog(QDialog):
    """Settings dialog for adjusting analysis parameters."""

    def __init__(self, parent=None):
        """Initialize the settings dialog."""
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 400, 300)
        self.init_ui()
        self.settings = {}

    def init_ui(self):
        """Initialize UI elements."""
        layout = QVBoxLayout()

        # Canny edge detection thresholds
        layout.addWidget(QLabel("Canny Edge Detection:"))
        
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(QLabel("Lower Threshold:"))
        self.lower_threshold = QSpinBox()
        self.lower_threshold.setRange(0, 255)
        self.lower_threshold.setValue(50)
        h_layout1.addWidget(self.lower_threshold)
        layout.addLayout(h_layout1)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Upper Threshold:"))
        self.upper_threshold = QSpinBox()
        self.upper_threshold.setRange(0, 255)
        self.upper_threshold.setValue(150)
        h_layout2.addWidget(self.upper_threshold)
        layout.addLayout(h_layout2)

        # Morphological operations
        layout.addWidget(QLabel("\nMorphological Operations:"))
        h_layout3 = QHBoxLayout()
        h_layout3.addWidget(QLabel("Iterations:"))
        self.morph_iterations = QSpinBox()
        self.morph_iterations.setRange(1, 10)
        self.morph_iterations.setValue(2)
        h_layout3.addWidget(self.morph_iterations)
        layout.addLayout(h_layout3)

        # Hough line detection
        layout.addWidget(QLabel("\nHough Line Detection:"))
        h_layout4 = QHBoxLayout()
        h_layout4.addWidget(QLabel("Min Line Length:"))
        self.min_line_length = QSpinBox()
        self.min_line_length.setRange(10, 200)
        self.min_line_length.setValue(30)
        h_layout4.addWidget(self.min_line_length)
        layout.addLayout(h_layout4)

        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_settings(self) -> dict:
        """Get current settings."""
        return {
            'lower_threshold': self.lower_threshold.value(),
            'upper_threshold': self.upper_threshold.value(),
            'morph_iterations': self.morph_iterations.value(),
            'min_line_length': self.min_line_length.value(),
        }
