$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

& $Python -m PyInstaller `
  --onefile `
  --console `
  --name metin2_automation_tool `
  --add-data "config.yaml;." `
  main.py

Write-Host "EXE hazir: $ProjectRoot\dist\metin2_automation_tool.exe"
