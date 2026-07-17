"""Main application window."""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFileDialog, QSplitter, QTextEdit, QStatusBar
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox

from src.image_processor import ImageProcessor
from src.fracture_analyzer import FractureAnalyzer
from src.ui.visualizer import ImageVisualizer
from src.ui.dialogs import SettingsDialog
from src.utils.image_export import ImageExporter


class ImageLabel(QLabel):
    """Custom label for displaying images."""

    def __init__(self):
        """Initialize the label."""
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_image(self, qimage: QImage):
        """Set and display an image.
        
        Args:
            qimage: QImage to display
        """
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaledToWidth(self.width(), Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Glass Fracture Analyzer")
        self.setGeometry(100, 100, 1400, 900)

        self.image_processor = None
        self.fracture_analyzer = FractureAnalyzer()
        self.current_image_path = None
        self.settings = {}

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top button panel
        button_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        button_layout.addWidget(load_btn)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.analyze_image)
        button_layout.addWidget(analyze_btn)

        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        button_layout.addWidget(export_btn)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.show_settings)
        button_layout.addWidget(settings_btn)

        main_layout.addLayout(button_layout)

        # Splitter for image display and metrics
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Image display area
        image_layout = QVBoxLayout()
        image_layout.addWidget(QLabel("Original Image:"))
        self.original_label = ImageLabel()
        image_layout.addWidget(self.original_label)
        
        image_layout.addWidget(QLabel("Processed Image:"))
        self.processed_label = ImageLabel()
        image_layout.addWidget(self.processed_label)

        image_widget = QWidget()
        image_widget.setLayout(image_layout)
        splitter.addWidget(image_widget)

        # Metrics and info panel
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(QLabel("Analysis Results:"))
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumWidth(300)
        metrics_layout.addWidget(self.metrics_text)

        metrics_widget = QWidget()
        metrics_widget.setLayout(metrics_layout)
        splitter.addWidget(metrics_widget)

        splitter.setSizes([900, 300])
        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "JPEG Images (*.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            try:
                self.image_processor = ImageProcessor(file_path)
                self.current_image_path = file_path
                
                # Display original image
                original = self.image_processor.get_original()
                qimage = ImageVisualizer.numpy_to_qimage(original)
                self.original_label.set_image(qimage)
                
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def analyze_image(self):
        """Analyze the loaded image."""
        if self.image_processor is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        try:
            # Detect edges
            edges = self.image_processor.detect_edges()
            
            # Clean up edges
            cleaned_edges = self.image_processor.apply_morphological_ops(edges)
            
            # Find cracks
            self.fracture_analyzer.find_contours(cleaned_edges)
            self.fracture_analyzer.extract_crack_lines(cleaned_edges)
            
            # Classify fracture type
            image_shape = self.image_processor.get_original().shape
            self.fracture_analyzer.classify_fracture_type(cleaned_edges, image_shape)
            
            # Display processed image with cracks
            original = self.image_processor.get_original()
            result = ImageVisualizer.draw_cracks_on_image(
                original, 
                self.fracture_analyzer.cracks
            )
            result = ImageVisualizer.draw_edges(cleaned_edges, result, alpha=0.3)
            
            qimage = ImageVisualizer.numpy_to_qimage(result)
            self.processed_label.set_image(qimage)
            
            # Display metrics
            metrics = self.fracture_analyzer.get_analysis_metrics()
            self.display_metrics(metrics)
            
            self.status_bar.showMessage("Analysis complete")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def display_metrics(self, metrics: dict):
        """Display analysis metrics in the text widget.
        
        Args:
            metrics: Dictionary of metrics to display
        """
        text = "Analysis Results:\n" + "="*40 + "\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                text += f"{key}: {value:.2f}\n"
            else:
                text += f"{key}: {value}\n"
        
        self.metrics_text.setText(text)

    def export_results(self):
        """Export analysis results."""
        if self.image_processor is None or not self.fracture_analyzer.cracks:
            QMessageBox.warning(self, "Warning", "Please analyze an image first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "JPEG Images (*.jpg);;PNG Images (*.png)"
        )
        
        if file_path:
            try:
                original = self.image_processor.get_original()
                result = ImageVisualizer.draw_cracks_on_image(
                    original,
                    self.fracture_analyzer.cracks
                )
                
                exporter = ImageExporter()
                exporter.save_image(file_path, result)
                
                self.status_bar.showMessage(f"Exported to: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", "Results exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.settings = dialog.get_settings()
            self.status_bar.showMessage("Settings updated")
