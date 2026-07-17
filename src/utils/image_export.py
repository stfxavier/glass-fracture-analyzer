"""Image export and file handling utilities."""

import cv2
import numpy as np
from pathlib import Path


class ImageExporter:
    """Handles image export and saving."""

    @staticmethod
    def save_image(file_path: str, image: np.ndarray, quality: int = 95) -> bool:
        """Save image to disk.
        
        Args:
            file_path: Path to save the image
            image: Image array in BGR format
            quality: JPEG quality (1-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save with appropriate format
            if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif file_path.lower().endswith('.png'):
                cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(file_path, image)
            
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    @staticmethod
    def create_comparison_image(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Create a side-by-side comparison image.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            Combined comparison image
        """
        # Ensure same height
        if original.shape[0] != processed.shape[0]:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # Create side-by-side image
        comparison = np.hstack((original, processed))
        return comparison
