"""Fracture pattern analysis and measurement."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class CrackInfo:
    """Information about a detected crack."""
    length: float
    angle: float  # In degrees
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    curvature: float


class FractureAnalyzer:
    """Analyzes fracture patterns in images."""

    def __init__(self):
        """Initialize the analyzer."""
        self.contours = []
        self.cracks = []
        self.fracture_type = None

    def find_contours(self, edge_map: np.ndarray) -> List[np.ndarray]:
        """Find contours in the edge map.
        
        Args:
            edge_map: Binary edge detection result
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(edge_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Filter by size
        min_length = 50
        self.contours = [c for c in contours if cv2.arcLength(c, False) > min_length]
        return self.contours

    def extract_crack_lines(self, edge_map: np.ndarray, min_length: int = 30) -> List[CrackInfo]:
        """Extract individual crack lines from edge map using Hough transform.
        
        Args:
            edge_map: Binary edge detection result
            min_length: Minimum crack length in pixels
            
        Returns:
            List of CrackInfo objects
        """
        # Use probabilistic Hough line transform
        lines = cv2.HoughLinesP(edge_map, rho=1, theta=np.pi/180, threshold=50,
                               minLineLength=min_length, maxLineGap=10)
        
        self.cracks = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                crack = self._create_crack_info(x1, y1, x2, y2)
                self.cracks.append(crack)
        
        return self.cracks

    def _create_crack_info(self, x1: int, y1: int, x2: int, y2: int) -> CrackInfo:
        """Create CrackInfo object from line coordinates."""
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        return CrackInfo(
            length=length,
            angle=angle,
            start_point=(x1, y1),
            end_point=(x2, y2),
            curvature=0.0  # Will be calculated for curved cracks
        )

    def classify_fracture_type(self, edge_map: np.ndarray, image_shape: Tuple[int, int]) -> str:
        """Classify the type of fracture pattern.
        
        Args:
            edge_map: Binary edge detection result
            image_shape: Shape of the original image
            
        Returns:
            Fracture type: 'hertzian', 'radial', 'concentric', or 'mixed'
        """
        height, width = image_shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate statistics about crack distribution
        if len(self.cracks) == 0:
            self.fracture_type = 'unknown'
            return self.fracture_type
        
        angles = [crack.angle for crack in self.cracks]
        angle_variance = np.var(angles)
        
        # Estimate center of fracture
        center_estimate = self._estimate_fracture_center()
        
        # Check for radial pattern (cracks originating from center)
        radial_count = sum(1 for crack in self.cracks 
                          if self._is_radial_crack(crack, center_estimate))
        radial_ratio = radial_count / len(self.cracks) if self.cracks else 0
        
        # Classification logic
        if radial_ratio > 0.6:
            self.fracture_type = 'radial'
        elif angle_variance < 100:
            self.fracture_type = 'hertzian'
        else:
            self.fracture_type = 'mixed'
        
        return self.fracture_type

    def _estimate_fracture_center(self) -> Tuple[float, float]:
        """Estimate the center of the fracture."""
        if not self.cracks:
            return (0, 0)
        
        all_x = []
        all_y = []
        for crack in self.cracks:
            all_x.extend([crack.start_point[0], crack.end_point[0]])
            all_y.extend([crack.start_point[1], crack.end_point[1]])
        
        return (np.mean(all_x), np.mean(all_y))

    def _is_radial_crack(self, crack: CrackInfo, center: Tuple[float, float]) -> bool:
        """Check if a crack is radial (pointing toward/from center)."""
        x_c, y_c = center
        x1, y1 = crack.start_point
        x2, y2 = crack.end_point
        
        # Vector from center to crack midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        vec_to_center = (mid_x - x_c, mid_y - y_c)
        
        # Crack direction vector
        crack_vec = (x2 - x1, y2 - y1)
        
        # Normalize
        len1 = np.sqrt(vec_to_center[0]**2 + vec_to_center[1]**2)
        len2 = np.sqrt(crack_vec[0]**2 + crack_vec[1]**2)
        
        if len1 < 1 or len2 < 1:
            return False
        
        vec_to_center = (vec_to_center[0]/len1, vec_to_center[1]/len1)
        crack_vec = (crack_vec[0]/len2, crack_vec[1]/len2)
        
        # Dot product (should be close to 1 or -1 for radial)
        dot_product = abs(vec_to_center[0] * crack_vec[0] + vec_to_center[1] * crack_vec[1])
        
        return dot_product > 0.6

    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get summary metrics of the fracture analysis.
        
        Returns:
            Dictionary with analysis metrics
        """
        if not self.cracks:
            return {
                'total_cracks': 0,
                'total_crack_length': 0,
                'average_crack_length': 0,
                'fracture_type': 'unknown',
                'fracture_area': 0
            }
        
        lengths = [crack.length for crack in self.cracks]
        total_length = sum(lengths)
        
        return {
            'total_cracks': len(self.cracks),
            'total_crack_length': total_length,
            'average_crack_length': total_length / len(self.cracks),
            'max_crack_length': max(lengths),
            'min_crack_length': min(lengths),
            'fracture_type': self.fracture_type or 'unknown',
        }
