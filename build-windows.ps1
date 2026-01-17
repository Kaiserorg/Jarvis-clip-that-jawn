# Build script for building Windows executables with PyInstaller
# Meant to be run on Windows (CI or local dev)

# Exit on errors
$ErrorActionPreference = 'Stop'

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Build both entrypoints; include model/ directory as data
pyinstaller --noconfirm --onefile --add-data "model;model" --name ScreenClipper screen_clipper.py
pyinstaller --noconfirm --onefile --add-data "model;model" --name record_wake_ui record_wake_ui.py

# Ensure dist contains the expected executables
if (!(Test-Path .\dist\ScreenClipper.exe)) {
    Write-Error "ScreenClipper.exe not found in dist"
}
if (!(Test-Path .\dist\record_wake_ui.exe)) {
    Write-Warning "record_wake_ui.exe not found in dist (it may have failed to build)"
}

Write-Host "Build complete. Artifacts are in dist\\"