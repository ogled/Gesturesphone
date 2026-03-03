param(
  [string]$PythonExe = ".\venv\Scripts\python.exe",
  [string]$AppName = "GesturesphoneRuntime"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path ".\Train\model.onnx")) {
  throw "Missing .\Train\model.onnx. Run: python Train/export_to_onnx.py"
}

if (-not (Test-Path ".\Train\model.runtime.json")) {
  throw "Missing .\Train\model.runtime.json. Run: python Train/export_to_onnx.py"
}

& $PythonExe -m pip install pyinstaller

& $PythonExe -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --name $AppName `
  --collect-all mediapipe `
  --paths ".\Backend" `
  --exclude-module torch `
  --exclude-module torchvision `
  --exclude-module torchaudio `
  --exclude-module pandas `
  --exclude-module scipy `
  --exclude-module tensorboard `
  --add-data ".\Frontend\dist;Frontend\dist" `
  --add-data ".\Train\model.onnx;Train" `
  --add-data ".\Train\model.runtime.json;Train" `
  --add-data ".\Backend\Config;Config" `
  ".\Backend\server.py"

if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Build completed: .\dist\$AppName"
