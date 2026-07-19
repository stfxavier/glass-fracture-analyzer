"""PDF report generation for fracture analysis."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class PDFReportGenerator:
    """Generates PDF reports from fracture analysis results."""

    def __init__(self):
        """Initialize the PDF report generator."""
        if not HAS_REPORTLAB:
            raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")

    @staticmethod
    def generate_report(output_path: str, original_image_path: str, processed_image_path: str,
                       analysis_metrics: Dict[str, Any], advanced_features: Dict[str, Any],
                       image_processor: Any = None) -> bool:
        """Generate a comprehensive PDF report.
        
        Args:
            output_path: Path to save the PDF report
            original_image_path: Path to original image (temporary file)
            processed_image_path: Path to processed image (temporary file)
            analysis_metrics: Dictionary of analysis metrics
            advanced_features: Dictionary of advanced feature detections
            image_processor: Reference to image processor for additional info
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create document
            doc = SimpleDocTemplate(output_path, pagesize=letter,
                                   rightMargin=0.5*inch, leftMargin=0.5*inch,
                                   topMargin=0.75*inch, bottomMargin=0.75*inch)
            
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2e5c8a'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Title
            story.append(Paragraph("Glass Fracture Analysis Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            story.append(Paragraph(f"Report Generated: {timestamp}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Images section
            story.append(Paragraph("Fracture Surface Images", heading_style))
            
            images_data = []
            try:
                if os.path.exists(original_image_path):
                    images_data.append(('Original Image', original_image_path))
                if os.path.exists(processed_image_path):
                    images_data.append(('Processed Image with Detected Features', processed_image_path))
            except:
                pass
            
            # Create image table
            if images_data:
                img_table_data = []
                for label, img_path in images_data:
                    try:
                        img = Image(img_path, width=3*inch, height=2.25*inch)
                        img_table_data.append([Paragraph(label, styles['Normal']), img])
                    except:
                        pass
                
                if img_table_data:
                    img_table = Table(img_table_data, colWidths=[1.5*inch, 3.5*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    story.append(img_table)
                    story.append(Spacer(1, 0.3*inch))
            
            # Analysis Metrics section
            story.append(PageBreak())
            story.append(Paragraph("Analysis Metrics", heading_style))
            
            metrics_data = [['Metric', 'Value']]
            for key, value in analysis_metrics.items():
                if isinstance(value, float):
                    metrics_data.append([str(key).replace('_', ' ').title(), f"{value:.2f}"])
                else:
                    metrics_data.append([str(key).replace('_', ' ').title(), str(value)])
            
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Advanced Features section
            if advanced_features:
                story.append(Paragraph("Detected Fracture Features", heading_style))
                
                features_data = [['Feature Type', 'Count', 'Avg. Confidence', 'Total Area']]
                for feature_name, feature_info in advanced_features.items():
                    if isinstance(feature_info, dict) and 'count' in feature_info:
                        features_data.append([
                            feature_name,
                            str(feature_info.get('count', 0)),
                            f"{feature_info.get('average_confidence', 0):.2f}",
                            f"{feature_info.get('total_area', 0):.0f}"
                        ])
                
                if len(features_data) > 1:
                    features_table = Table(features_data, colWidths=[2*inch, 1*inch, 1.5*inch, 1.5*inch])
                    features_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                    ]))
                    story.append(features_table)
                    story.append(Spacer(1, 0.3*inch))
            
            # Conclusions section
            story.append(PageBreak())
            story.append(Paragraph("Analysis Summary", heading_style))
            
            # Generate summary text
            if analysis_metrics.get('total_cracks', 0) > 0:
                summary_text = f"""
                This fracture analysis detected {analysis_metrics.get('total_cracks', 0)} major crack(s) 
                with a total length of {analysis_metrics.get('total_crack_length', 0):.2f} pixels. 
                The fracture was classified as <b>{analysis_metrics.get('fracture_type', 'unknown')}</b> type.
                <br/><br/>
                The average crack length is {analysis_metrics.get('average_crack_length', 0):.2f} pixels, 
                with the longest crack measuring {analysis_metrics.get('max_crack_length', 0):.2f} pixels.
                """
            else:
                summary_text = "No significant fractures were detected in this image."
            
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Footer
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(
                "<i>This report was automatically generated by Glass Fracture Analyzer software.</i>",
                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
            ))
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return False
