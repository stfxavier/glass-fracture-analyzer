"""Unit tests for Glass Fracture Analyzer."""

import unittest
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path

from src.image_processor import ImageProcessor
from src.fracture_analyzer import FractureAnalyzer, CrackInfo
from src.advanced_fracture_analyzer import AdvancedFractureAnalyzer, FractureFeatureType
from src.utils.image_export import ImageExporter


class TestImageProcessor(unittest.TestCase):
    """Test cases for ImageProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        
        # Create a synthetic test image with a crack-like pattern
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:, :] = 200  # Light gray background
        # Add a diagonal line (simulating a crack)
        cv2.line(test_img, (100, 100), (500, 400), (50, 50, 50), 3)
        cv2.imwrite(self.test_image_path, test_img)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        os.rmdir(self.temp_dir)

    def test_image_loading(self):
        """Test that images are loaded correctly."""
        processor = ImageProcessor(self.test_image_path)
        self.assertIsNotNone(processor.original_image)
        self.assertEqual(processor.original_image.shape, (480, 640, 3))

    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        processor = ImageProcessor(self.test_image_path)
        gray = processor.get_gray()
        self.assertEqual(len(gray.shape), 2)  # Should be 2D
        self.assertEqual(gray.shape, (480, 640))

    def test_edge_detection(self):
        """Test edge detection produces valid output."""
        processor = ImageProcessor(self.test_image_path)
        edges = processor.detect_edges()
        self.assertEqual(edges.shape, processor.get_gray().shape)
        self.assertTrue(np.any(edges > 0))  # Should detect some edges

    def test_bilateral_filter(self):
        """Test bilateral filtering."""
        processor = ImageProcessor(self.test_image_path)
        filtered = processor.apply_bilateral_filter()
        self.assertEqual(filtered.shape, processor.get_gray().shape)
        self.assertIsNotNone(filtered)

    def test_image_dimensions(self):
        """Test image dimension retrieval."""
        processor = ImageProcessor(self.test_image_path)
        width, height = processor.get_image_dimensions()
        self.assertEqual((width, height), (640, 480))


class TestFractureAnalyzer(unittest.TestCase):
    """Test cases for FractureAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FractureAnalyzer()
        # Create synthetic edge map with a line
        self.edge_map = np.zeros((300, 400), dtype=np.uint8)
        cv2.line(self.edge_map, (50, 50), (350, 250), 255, 2)

    def test_crack_info_creation(self):
        """Test CrackInfo dataclass."""
        crack = CrackInfo(
            length=100.0,
            angle=45.0,
            start_point=(0, 0),
            end_point=(100, 100),
            curvature=0.01
        )
        self.assertEqual(crack.length, 100.0)
        self.assertEqual(crack.angle, 45.0)

    def test_contour_detection(self):
        """Test contour detection."""
        contours = self.analyzer.find_contours(self.edge_map)
        self.assertIsInstance(contours, list)

    def test_crack_line_extraction(self):
        """Test crack line extraction using Hough transform."""
        cracks = self.analyzer.extract_crack_lines(self.edge_map)
        self.assertIsInstance(cracks, list)
        if len(cracks) > 0:
            self.assertIsInstance(cracks[0], CrackInfo)

    def test_fracture_type_classification(self):
        """Test fracture type classification."""
        self.analyzer.extract_crack_lines(self.edge_map)
        fracture_type = self.analyzer.classify_fracture_type(self.edge_map, (400, 300, 3))
        self.assertIn(fracture_type, ['hertzian', 'radial', 'concentric', 'mixed', 'unknown'])

    def test_analysis_metrics(self):
        """Test analysis metrics generation."""
        self.analyzer.extract_crack_lines(self.edge_map)
        metrics = self.analyzer.get_analysis_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_cracks', metrics)
        self.assertIn('fracture_type', metrics)


class TestAdvancedFractureAnalyzer(unittest.TestCase):
    """Test cases for AdvancedFractureAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.advanced_analyzer = AdvancedFractureAnalyzer()
        # Create synthetic test image
        self.test_image = np.ones((300, 400), dtype=np.uint8) * 150
        # Add some structure
        cv2.circle(self.test_image, (200, 150), 50, 100, -1)
        
        # Create edge map
        self.edges = cv2.Canny(self.test_image, 50, 150)

    def test_advanced_analyzer_initialization(self):
        """Test that advanced analyzer initializes correctly."""
        self.assertIsNotNone(self.advanced_analyzer.features)
        self.assertIsInstance(self.advanced_analyzer.features, list)

    def test_texture_analysis(self):
        """Test texture property analysis."""
        self.advanced_analyzer._analyze_texture_properties(self.test_image)
        self.assertIn('variance_map', self.advanced_analyzer.texture_analysis)
        self.assertIn('mean_smoothness', self.advanced_analyzer.texture_analysis)

    def test_feature_detection(self):
        """Test that feature detection runs without error."""
        features = self.advanced_analyzer.analyze_fracture_surface(self.test_image, self.edges)
        self.assertIsInstance(features, dict)
        # Check that feature types are properly mapped
        for feature_type in FractureFeatureType:
            if feature_type in features:
                self.assertIsInstance(features[feature_type], list)

    def test_feature_summary(self):
        """Test feature summary generation."""
        self.advanced_analyzer.analyze_fracture_surface(self.test_image, self.edges)
        summary = self.advanced_analyzer.get_feature_summary()
        self.assertIsInstance(summary, dict)

    def test_mirror_detection(self):
        """Test mirror region detection."""
        features = self.advanced_analyzer._detect_mirror(self.test_image, self.edges)
        self.assertIsInstance(features, list)

    def test_hackle_detection(self):
        """Test hackle region detection."""
        features = self.advanced_analyzer._detect_hackle(self.test_image, self.edges)
        self.assertIsInstance(features, list)


class TestImageExporter(unittest.TestCase):
    """Test cases for ImageExporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[:, :] = [100, 150, 200]

    def tearDown(self):
        """Clean up test fixtures."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_save_jpg_image(self):
        """Test saving image as JPG."""
        save_path = os.path.join(self.temp_dir, 'test.jpg')
        result = ImageExporter.save_image(save_path, self.test_image)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(save_path))

    def test_save_png_image(self):
        """Test saving image as PNG."""
        save_path = os.path.join(self.temp_dir, 'test.png')
        result = ImageExporter.save_image(save_path, self.test_image)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(save_path))

    def test_comparison_image_creation(self):
        """Test side-by-side comparison image creation."""
        image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2[:, :] = [50, 100, 150]
        comparison = ImageExporter.create_comparison_image(self.test_image, image2)
        self.assertEqual(comparison.shape[0], 100)  # Height should match
        self.assertEqual(comparison.shape[1], 200)  # Width should be doubled


class TestPDFReportGenerator(unittest.TestCase):
    """Test cases for PDFReportGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Create test images
        self.original_image_path = os.path.join(self.temp_dir, 'original.jpg')
        self.processed_image_path = os.path.join(self.temp_dir, 'processed.jpg')
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 150
        cv2.imwrite(self.original_image_path, test_img)
        cv2.imwrite(self.processed_image_path, test_img)
        
        self.output_path = os.path.join(self.temp_dir, 'report.pdf')

    def tearDown(self):
        """Clean up test fixtures."""
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)

    def test_pdf_generation(self):
        """Test PDF report generation."""
        try:
            from src.utils.pdf_reporter import PDFReportGenerator
            
            test_metrics = {
                'total_cracks': 3,
                'total_crack_length': 500.0,
                'average_crack_length': 166.67,
                'fracture_type': 'radial'
            }
            
            test_features = {
                'Mirror': {'count': 1, 'average_confidence': 0.85, 'total_area': 5000},
                'Hackle': {'count': 2, 'average_confidence': 0.65, 'total_area': 3000}
            }
            
            result = PDFReportGenerator.generate_report(
                self.output_path,
                self.original_image_path,
                self.processed_image_path,
                test_metrics,
                test_features
            )
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(self.output_path))
        except ImportError:
            self.skipTest("reportlab not installed")


if __name__ == '__main__':
    unittest.main()
