"""Image visualization and display utilities."""

import cv2
import numpy as np
from typing import Tuple
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont
from PyQt6.QtCore import Qt


class ImageVisualizer:
    """Handles visualization of images and analysis results."""

    @staticmethod
    def numpy_to_qimage(cv_image: np.ndarray) -> QImage:
        """Convert OpenCV image to QImage for PyQt6 display.
        
        Args:
            cv_image: Image in OpenCV format (BGR or grayscale)
            
        Returns:
            QImage for display in PyQt6 widgets
        """
        if len(cv_image.shape) == 2:  # Grayscale
            height, width = cv_image.shape
            bytes_per_line = width
            converted = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:  # Color (BGR)
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            converted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        q_img = QImage(converted.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return q_img

    @staticmethod
    def draw_cracks_on_image(image: np.ndarray, cracks: list, color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 2) -> np.ndarray:
        """Draw detected cracks on the image.
        
        Args:
            image: Original image
            cracks: List of CrackInfo objects
            color: Color in BGR format
            thickness: Line thickness
            
        Returns:
            Image with drawn cracks
        """
        result = image.copy()
        for crack in cracks:
            pt1 = crack.start_point
            pt2 = crack.end_point
            cv2.line(result, pt1, pt2, color, thickness)
            # Draw endpoints
            cv2.circle(result, pt1, 3, (255, 0, 0), -1)
            cv2.circle(result, pt2, 3, (0, 0, 255), -1)
        
        return result

    @staticmethod
    def draw_edges(edges: np.ndarray, original: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay edges on original image.
        
        Args:
            edges: Binary edge map
            original: Original image
            alpha: Opacity of overlay
            
        Returns:
            Overlaid image
        """
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[edges > 0] = [0, 255, 0]  # Green for edges
        
        result = cv2.addWeighted(original, 1 - alpha, edges_colored, alpha, 0)
        return result

    @staticmethod
    def add_text_info(image: np.ndarray, metrics: dict) -> np.ndarray:
        """Add analysis metrics as text overlay.
        
        Args:
            image: Image to annotate
            metrics: Dictionary of metrics to display
            
        Returns:
            Annotated image
        """
        result = image.copy()
        y_offset = 30
        
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(result, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return result
