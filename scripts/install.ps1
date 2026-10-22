$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-Checked {
  param([string]$Program, [string[]]$Arguments)
  & $Program @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Program : code $LASTEXITCODE" }
}
if (-not (Test-Path '.venv/Scripts/python.exe')) {
  Invoke-Checked 'py' @('-3.12', '-m', 'venv', '.venv')
}
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
Invoke-Checked $python @('-m', 'pip', 'install', '-r', 'requirements.txt')
# Trackers declares opencv-python; use the already installed contrib distribution.
Invoke-Checked $python @('-m', 'pip', 'install', '--no-deps', 'trackers==2.6.1', 'rapidocr-onnxruntime==1.4.4')
Push-Location 'frontend'
try {
  Invoke-Checked 'npm.cmd' @('ci')
  Invoke-Checked 'npm.cmd' @('run', 'build')
} finally { Pop-Location }
Write-Host 'Installation terminée. Ouvrez DEMARRER.bat.'
