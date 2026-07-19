"""Advanced fracture feature recognition and classification."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FractureFeatureType(Enum):
    """Types of fracture features that can be detected."""
    MIRROR = "Mirror"
    MIST = "Mist"
    HACKLE = "Hackle"
    MIRROR_MIST_BOUNDARY = "Mirror-Mist Boundary"
    VELOCITY_HACKLE = "Velocity Hackle"
    EYELASH_HACKLE = "Eyelash Hackle"
    WALLNER_LINE = "Wallner Line"
    CANTILEVER_CURL = "Cantilever Curl"
    WAKE_HACKLE = "Wake Hackle"


@dataclass
class FractureFeature:
    """Information about a detected fracture feature."""
    feature_type: FractureFeatureType
    location: Tuple[int, int]  # Center point
    area: float
    confidence: float  # 0-1 confidence score
    properties: Dict[str, float]  # Feature-specific properties


class AdvancedFractureAnalyzer:
    """Advanced analyzer for detecting specific fracture features."""

    def __init__(self):
        """Initialize the advanced analyzer."""
        self.features: List[FractureFeature] = []
        self.feature_map = None
        self.texture_analysis = {}

    def analyze_fracture_surface(self, image: np.ndarray, edges: np.ndarray) -> Dict[FractureFeatureType, List[FractureFeature]]:
        """Comprehensive analysis of fracture surface features.
        
        Args:
            image: Original image in grayscale
            edges: Edge detection result
            
        Returns:
            Dictionary mapping feature types to detected features
        """
        features_by_type = {}
        
        # Analyze different regions
        self._analyze_texture_properties(image)
        
        # Detect each type of feature
        features_by_type[FractureFeatureType.MIRROR] = self._detect_mirror(image, edges)
        features_by_type[FractureFeatureType.MIST] = self._detect_mist(image, edges)
        features_by_type[FractureFeatureType.HACKLE] = self._detect_hackle(image, edges)
        features_by_type[FractureFeatureType.MIRROR_MIST_BOUNDARY] = self._detect_mirror_mist_boundary(image, edges)
        features_by_type[FractureFeatureType.VELOCITY_HACKLE] = self._detect_velocity_hackle(image, edges)
        features_by_type[FractureFeatureType.EYELASH_HACKLE] = self._detect_eyelash_hackle(image, edges)
        features_by_type[FractureFeatureType.WALLNER_LINE] = self._detect_wallner_lines(image, edges)
        features_by_type[FractureFeatureType.CANTILEVER_CURL] = self._detect_cantilever_curl(image, edges)
        features_by_type[FractureFeatureType.WAKE_HACKLE] = self._detect_wake_hackle(image, edges)
        
        # Flatten and store all features
        self.features = []
        for feature_list in features_by_type.values():
            self.features.extend(feature_list)
        
        return features_by_type

    def _analyze_texture_properties(self, image: np.ndarray) -> None:
        """Analyze texture properties of the fracture surface.
        
        Args:
            image: Grayscale image
        """
        # Calculate local variance (smoothness indicator)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mean = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        variance = cv2.morphologyEx((image - mean) ** 2, cv2.MORPH_OPEN, kernel)
        
        self.texture_analysis['variance_map'] = variance
        self.texture_analysis['mean_smoothness'] = np.mean(variance)
        self.texture_analysis['smoothness_std'] = np.std(variance)

    def _detect_mirror(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect mirror region (smooth, reflective fracture surface).
        
        Mirror appears as smooth, uniform region with low texture variance.
        Typically found near the fracture origin.
        """
        features = []
        variance_map = self.texture_analysis.get('variance_map', np.zeros_like(image))
        
        if variance_map.size == 0:
            return features
        
        # Smooth regions have low variance
        smoothness = 255 - cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Find highly smooth regions
        mirror_mask = cv2.threshold(smoothness, 200, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(mirror_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Calculate confidence based on smoothness
                    mask = np.zeros_like(mirror_mask)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    confidence = np.mean(smoothness[mask > 0]) / 255.0
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.MIRROR,
                        location=(cx, cy),
                        area=area,
                        confidence=min(confidence, 1.0),
                        properties={
                            'smoothness_score': float(np.mean(smoothness[mask > 0])),
                            'perimeter': float(cv2.arcLength(contour, True)),
                            'circularity': 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2 + 1e-5)
                        }
                    )
                    features.append(feature)
        
        return features

    def _detect_mist(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect mist region (intermediate texture with fine ripples).
        
        Mist appears as region with medium texture variation.
        """
        features = []
        variance_map = self.texture_analysis.get('variance_map', np.zeros_like(image))
        
        if variance_map.size == 0:
            return features
        
        # Normalize variance map
        normalized_variance = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Mist has medium variance (not too smooth, not too rough)
        mist_mask = cv2.inRange(normalized_variance, 80, 180)
        
        contours, _ = cv2.findContours(mist_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Minimum area threshold
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    mask = np.zeros_like(mist_mask)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    confidence = 0.7  # Medium confidence for mist detection
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.MIST,
                        location=(cx, cy),
                        area=area,
                        confidence=confidence,
                        properties={
                            'texture_variance': float(np.mean(normalized_variance[mask > 0])),
                            'edge_density': float(np.sum(edges[mask > 0]) / area) if area > 0 else 0.0,
                        }
                    )
                    features.append(feature)
        
        return features

    def _detect_hackle(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect hackle region (rough, multiple micro-fractures).
        
        Hackle appears as rough region with high edge density.
        """
        features = []
        
        # Detect rough regions (high edge density)
        edge_density = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        
        # Find regions with high edge concentration
        _, hackle_mask = cv2.threshold(edge_density, 100, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(hackle_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 400:  # Minimum area threshold
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    mask = np.zeros_like(hackle_mask)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    edge_count = np.sum(edges[mask > 0])
                    confidence = min(edge_count / (area * 2), 1.0)
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.HACKLE,
                        location=(cx, cy),
                        area=area,
                        confidence=confidence,
                        properties={
                            'edge_density': float(edge_count / area) if area > 0 else 0.0,
                            'roughness': float(np.std(image[mask > 0])),
                        }
                    )
                    features.append(feature)
        
        return features

    def _detect_mirror_mist_boundary(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect mirror-mist boundary (transition zone).
        
        Boundary appears as distinct contour between smooth and textured regions.
        """
        features = []
        
        # Find edges that represent transitions
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        boundary_mask = cv2.threshold(np.abs(laplacian), 50, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        
        contours, _ = cv2.findContours(boundary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            arc_length = cv2.arcLength(contour, True)
            if arc_length > 100:  # Minimum length threshold
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.MIRROR_MIST_BOUNDARY,
                        location=(cx, cy),
                        area=arc_length,
                        confidence=0.75,
                        properties={
                            'boundary_length': float(arc_length),
                            'boundary_curvature': float(arc_length / cv2.contourArea(contour) + 1e-5),
                        }
                    )
                    features.append(feature)
        
        return features

    def _detect_velocity_hackle(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect velocity hackle (rapidly propagating fracture).
        
        Appears as parallel ridge patterns at high angles to crack direction.
        """
        features = []
        
        # Detect line-like patterns using morphological operations
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        
        # Look for parallel line patterns
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        velocity_pattern = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        contours, _ = cv2.findContours(velocity_pattern, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Fit ellipse to characterize pattern
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        (_, _), (major, minor), angle = ellipse
                        aspect_ratio = major / (minor + 1e-5)
                    else:
                        aspect_ratio = 1.0
                        angle = 0
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.VELOCITY_HACKLE,
                        location=(cx, cy),
                        area=area,
                        confidence=0.65 * min(aspect_ratio / 5.0, 1.0),
                        properties={
                            'aspect_ratio': float(aspect_ratio),
                            'orientation_angle': float(angle),
                        }
                    )
                    features.append(feature)
        
        return features

    def _detect_eyelash_hackle(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect eyelash hackle (fractured fragments at edge).
        
        Appears as short linear features radiating from main fracture.
        """
        features = []
        
        # Detect short, dense linear features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Look for branching patterns (fracture endpoints)
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            arc_length = cv2.arcLength(contour, True)
            
            if 100 < area < 2000:  # Specific size range for eyelash
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Eyelash has high perimeter-to-area ratio
                    perimeter_ratio = arc_length / (2 * np.sqrt(np.pi * area) + 1e-5)
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.EYELASH_HACKLE,
                        location=(cx, cy),
                        area=area,
                        confidence=min(0.8 * perimeter_ratio / 2.0, 1.0),
                        properties={
                            'perimeter_ratio': float(perimeter_ratio),
                            'compactness': float(4 * np.pi * area / (arc_length ** 2 + 1e-5)),
                        }
                    )
                    features.append(feature)
        
        return features

    def _detect_wallner_lines(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect Wallner lines (curved stress wave patterns).
        
        Appears as curved concentric patterns on fracture surface.
        """
        features = []
        
        # Apply Hough circle detection to find curved patterns
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=200)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                
                # Extract region and check for curved lines
                y1 = max(0, cy - r - 10)
                y2 = min(image.shape[0], cy + r + 10)
                x1 = max(0, cx - r - 10)
                x2 = min(image.shape[1], cx + r + 10)
                
                region = edges[y1:y2, x1:x2]
                if region.size > 0:
                    edge_concentration = np.sum(region) / region.size
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.WALLNER_LINE,
                        location=(cx, cy),
                        area=np.pi * r * r,
                        confidence=min(edge_concentration / 255.0, 1.0),
                        properties={
                            'radius': float(r),
                            'edge_concentration': float(edge_concentration),
                        }
                    )
                    features.append(feature)
        
        return features

    def _detect_cantilever_curl(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect cantilever curl (final fracture feature).
        
        Appears as curved terminal feature where fracture propagation ended.
        """
        features = []
        
        # Find end points of fractures (high curvature regions)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            arc_length = cv2.arcLength(contour, False)
            if arc_length > 50:
                # Approximate contour to reduce noise
                epsilon = 0.02 * arc_length
                approx = cv2.approxPolyDP(contour, epsilon, False)
                
                # Calculate curvature
                if len(approx) >= 3:
                    # Check for curl-like features (high curvature at ends)
                    start_point = approx[0][0]
                    end_point = approx[-1][0]
                    
                    # Calculate curvature near endpoint
                    if len(approx) >= 4:
                        p1 = approx[-3][0]
                        p2 = approx[-2][0]
                        p3 = approx[-1][0]
                        
                        # Calculate curvature
                        curvature = self._calculate_point_curvature(p1, p2, p3)
                        
                        if curvature > 0.05:  # High curvature threshold
                            M = cv2.moments(contour)
                            if M['m00'] != 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                                
                                feature = FractureFeature(
                                    feature_type=FractureFeatureType.CANTILEVER_CURL,
                                    location=(cx, cy),
                                    area=cv2.contourArea(contour),
                                    confidence=min(0.9 * curvature / 0.1, 1.0),
                                    properties={
                                        'curvature': float(curvature),
                                        'contour_length': float(arc_length),
                                    }
                                )
                                features.append(feature)
        
        return features

    def _detect_wake_hackle(self, image: np.ndarray, edges: np.ndarray) -> List[FractureFeature]:
        """Detect wake hackle (disturbances behind main fracture).
        
        Appears as irregular patterns trailing the main crack direction.
        """
        features = []
        
        # Detect trailing patterns using directional analysis
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate local gradient direction changes
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # Find regions with significant direction changes (wake pattern)
        direction_change = cv2.Laplacian(direction, cv2.CV_64F)
        wake_mask = cv2.threshold(np.abs(direction_change), 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        
        contours, _ = cv2.findContours(wake_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 150:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Calculate confidence based on magnitude in region
                    mask = np.zeros_like(wake_mask)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    confidence = np.mean(magnitude[mask > 0]) / (np.max(magnitude) + 1e-5)
                    
                    feature = FractureFeature(
                        feature_type=FractureFeatureType.WAKE_HACKLE,
                        location=(cx, cy),
                        area=area,
                        confidence=min(confidence, 1.0),
                        properties={
                            'direction_variance': float(np.var(direction[mask > 0])),
                            'mean_magnitude': float(np.mean(magnitude[mask > 0])),
                        }
                    )
                    features.append(feature)
        
        return features

    @staticmethod
    def _calculate_point_curvature(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate curvature at a point given three consecutive points.
        
        Args:
            p1, p2, p3: Three consecutive points
            
        Returns:
            Curvature value
        """
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        
        # Calculate vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])
        
        # Normalize
        v1_len = np.linalg.norm(v1) + 1e-5
        v2_len = np.linalg.norm(v2) + 1e-5
        
        v1 = v1 / v1_len
        v2 = v2 / v2_len
        
        # Calculate angle between vectors
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product, -1, 1))
        
        # Curvature is angle / distance
        distance = v1_len + v2_len
        curvature = angle / (distance + 1e-5)
        
        return curvature

    def get_feature_summary(self) -> Dict[str, any]:
        """Get summary of detected features.
        
        Returns:
            Dictionary with feature statistics
        """
        summary = {}
        
        for feature_type in FractureFeatureType:
            features_of_type = [f for f in self.features if f.feature_type == feature_type]
            if features_of_type:
                summary[feature_type.value] = {
                    'count': len(features_of_type),
                    'average_confidence': np.mean([f.confidence for f in features_of_type]),
                    'total_area': sum([f.area for f in features_of_type]),
                }
        
        return summary
