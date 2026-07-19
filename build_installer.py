"""Windows installer creation script using PyInstaller."""

import os
import subprocess
import sys
from pathlib import Path


def create_installer():
    """Create a Windows installer for the Glass Fracture Analyzer."""
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Create dist directory if it doesn't exist
    dist_dir = Path('dist')
    dist_dir.mkdir(exist_ok=True)
    
    print("Building Windows executable...")
    
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--name=GlassFractureAnalyzer',
        '--onefile',
        '--windowed',  # No console window
        '--icon=assets/icon.ico',  # Optional: if you have an icon
        '--add-data=src:src',
        '--distpath=dist',
        '--buildpath=build',
        '--specpath=.',
        'src/main.py'
    ]
    
    # Run PyInstaller
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Executable built successfully")
        print(f"  Location: dist/GlassFractureAnalyzer.exe")
        
        # Create installer using NSIS if available
        try:
            create_nsis_installer()
        except Exception as e:
            print(f"NSIS installer creation skipped: {e}")
            print("\nYou can manually create an installer using NSIS with the executable.")
    else:
        print("✗ Build failed:")
        print(result.stderr)
        sys.exit(1)


def create_nsis_installer():
    """Create NSIS installer script."""
    
    nsis_script = ''';
; Glass Fracture Analyzer Installer
; Built with NSIS 3.x

!include "MUI2.nsh"

; Name and file
Name "Glass Fracture Analyzer"
OutFile "dist\\GlassFractureAnalyzer-Installer.exe"

; Default installation folder
InstallDir "$PROGRAMFILES\\GlassFractureAnalyzer"

; Get installation folder from registry if available
InstallDirRegKey HKCU "Software\\GlassFractureAnalyzer" ""

; Request application privileges
RequestExecutionLevel admin

; MUI Settings
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

; Installer sections
Section "Install"
  SetOutPath "$INSTDIR"
  File "dist\\GlassFractureAnalyzer.exe"
  
  ; Create shortcuts
  CreateDirectory "$SMPROGRAMS\\GlassFractureAnalyzer"
  CreateShortcut "$SMPROGRAMS\\GlassFractureAnalyzer\\Glass Fracture Analyzer.lnk" "$INSTDIR\\GlassFractureAnalyzer.exe"
  CreateShortcut "$SMPROGRAMS\\GlassFractureAnalyzer\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
  
  ; Create uninstaller
  WriteUninstaller "$INSTDIR\\Uninstall.exe"
  WriteRegStr HKCU "Software\\GlassFractureAnalyzer" "" "$INSTDIR"
SectionEnd

; Uninstaller section
Section "Uninstall"
  Delete "$INSTDIR\\GlassFractureAnalyzer.exe"
  Delete "$INSTDIR\\Uninstall.exe"
  RMDir "$INSTDIR"
  
  Delete "$SMPROGRAMS\\GlassFractureAnalyzer\\Glass Fracture Analyzer.lnk"
  Delete "$SMPROGRAMS\\GlassFractureAnalyzer\\Uninstall.lnk"
  RMDir "$SMPROGRAMS\\GlassFractureAnalyzer"
  
  DeleteRegKey /ifempty HKCU "Software\\GlassFractureAnalyzer"
SectionEnd
'''
    
    with open('GlassFractureAnalyzer.nsi', 'w') as f:
        f.write(nsis_script)
    
    print("\n✓ NSIS installer script created: GlassFractureAnalyzer.nsi")
    print("  To build the installer, install NSIS and run: makensis GlassFractureAnalyzer.nsi")


if __name__ == '__main__':
    create_installer()
