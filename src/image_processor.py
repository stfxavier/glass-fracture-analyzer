"""Image processing pipeline for fracture analysis."""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """Handles core image processing operations."""

    def __init__(self, image_path: str):
        """Initialize with an image path.
        
        Args:
            image_path: Path to the JPG image file
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = None
        self.edges = None

    def get_original(self) -> np.ndarray:
        """Get the original image in BGR format."""
        return self.original_image.copy()

    def get_gray(self) -> np.ndarray:
        """Get the grayscale version."""
        return self.gray_image.copy()

    def apply_clahe(self, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            clip_limit: Contrast limit threshold
            tile_size: Size of grid tiles
            
        Returns:
            Enhanced grayscale image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(self.gray_image)
        return enhanced

    def detect_edges(self, sigma: float = 1.0) -> np.ndarray:
        """Detect edges using Canny edge detection.
        
        Args:
            sigma: Gaussian blur sigma for preprocessing
            
        Returns:
            Binary edge map
        """
        # Enhance contrast
        enhanced = self.apply_clahe()
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Blur for edge detection
        blurred = cv2.GaussianBlur(denoised, (5, 5), sigma)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        self.edges = edges
        return edges.copy()

    def apply_morphological_ops(self, edges: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply morphological operations to clean up edges.
        
        Args:
            edges: Edge map (uses self.edges if None)
            
        Returns:
            Cleaned edge map
        """
        if edges is None:
            edges = self.edges if self.edges is not None else self.detect_edges()
        
        # Close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened

    def apply_bilateral_filter(self, diameter: int = 9, sigma_color: float = 75, 
                              sigma_space: float = 75) -> np.ndarray:
        """Apply bilateral filtering to preserve edges while smoothing.
        
        Args:
            diameter: Diameter of pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(self.gray_image, diameter, sigma_color, sigma_space)

    def get_image_dimensions(self) -> Tuple[int, int]:
        """Get image width and height."""
        height, width = self.original_image.shape[:2]
        return width, height
