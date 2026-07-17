# Glass Fracture Analyzer

A Windows desktop application for analyzing optical microscopy images of fractured glassware. The app detects fracture patterns, measures crack characteristics, and exports analysis results.

## Features

- **Image Loading**: Support for JPG images (1920x1080 resolution)
- **Fracture Detection**: 
  - Automatic crack pattern detection using edge detection
  - Classification of fracture types (Hertzian, radial, concentric)
  - Crack length and angle measurements
- **Visualization**:
  - Side-by-side comparison of original and processed images
  - Overlay of detected cracks with annotations
  - Real-time visualization updates
- **Export**:
  - Save processed images in JPG format
  - Export analysis metrics and statistics

## Requirements

- Python 3.8+
- Windows 10/11

## Installation

1. Clone the repository:
```bash
git clone https://github.com/stfxavier/glass-fracture-analyzer.git
cd glass-fracture-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python src/main.py
```

## Project Structure

```
src/
├── main.py                    # Application entry point
├── image_processor.py         # Image processing pipeline
├── fracture_analyzer.py       # Fracture detection and analysis
├── ui/
│   ├── main_window.py         # Main GUI window
│   ├── dialogs.py             # Dialog windows
│   └── visualizer.py          # Image visualization
└── utils/
    └── image_export.py        # Export functionality
```

## License

MIT License
