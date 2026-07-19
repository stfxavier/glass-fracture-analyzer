# Glass Fracture Analyzer - Build Instructions

## Requirements

- Python 3.8+
- All dependencies from `requirements.txt`
- PyInstaller (for building executables)
- NSIS 3.x (optional, for creating installer)

## Building the Application

### 1. Install Build Dependencies

```bash
pip install PyInstaller pyinstaller-hooks-contrib
```

### 2. Build Executable

```bash
python build_installer.py
```

This will create:
- `dist/GlassFractureAnalyzer.exe` - Standalone executable
- `GlassFractureAnalyzer.nsi` - NSIS installer script (optional)

### 3. (Optional) Create Windows Installer

If you have NSIS installed:

```bash
makensis GlassFractureAnalyzer.nsi
```

This creates `dist/GlassFractureAnalyzer-Installer.exe`

## Running Tests

```bash
python -m pytest tests/ -v
```

Or use unittest:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Building PDF Reports

To generate PDF reports, ensure reportlab is installed:

```bash
pip install reportlab
```

## Project Structure

```
glass-fracture-analyzer/
├── src/
│   ├── main.py                           # Application entry point
│   ├── image_processor.py                # Image processing
│   ├── fracture_analyzer.py              # Basic fracture analysis
│   ├── advanced_fracture_analyzer.py     # Advanced feature detection
│   ├── ui/
│   │   ├── main_window.py                # Main GUI window
│   │   ├── dialogs.py                    # Settings dialogs
│   │   └── visualizer.py                 # Image visualization
│   └── utils/
│       ├── image_export.py               # Image export
│       └── pdf_reporter.py               # PDF report generation
├── tests/
│   └── test_glass_fracture_analyzer.py  # Unit tests
├── build_installer.py                    # Build script
├── requirements.txt                      # Dependencies
└── README.md
```

## Features

### Basic Fracture Analysis
- Edge detection and crack identification
- Fracture type classification (Hertzian, Radial, Concentric, Mixed)
- Crack measurement (length, angle, position)

### Advanced Feature Detection
1. **Mirror** - Smooth, reflective fracture surface
2. **Mist** - Intermediate texture with fine ripples
3. **Hackle** - Rough region with micro-fractures
4. **Mirror-Mist Boundary** - Transition zone detection
5. **Velocity Hackle** - High-speed fracture patterns
6. **Eyelash Hackle** - Fractured edge fragments
7. **Wallner Lines** - Curved stress wave patterns
8. **Cantilever Curl** - Terminal fracture feature
9. **Wake Hackle** - Disturbances behind main fracture

### Export Options
- JPG/PNG image export with overlays
- PDF report generation with metrics and images
- Analysis metrics export

## Troubleshooting

### ImportError: No module named 'PyQt6'
Install PyQt6:
```bash
pip install PyQt6
```

### Build fails with icon error
Either provide an icon file at `assets/icon.ico` or remove the `--icon` parameter from the build script.

### PDF generation not working
Install reportlab:
```bash
pip install reportlab
```
