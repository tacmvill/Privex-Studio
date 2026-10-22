$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
& $python -m pip install 'pyinstaller==6.22.3'
if ($LASTEXITCODE -ne 0) { throw 'Installation de PyInstaller interrompue.' }
Push-Location frontend
try {
  & npm.cmd ci
  if ($LASTEXITCODE -ne 0) { throw 'npm ci a échoué.' }
  & npm.cmd run build
  if ($LASTEXITCODE -ne 0) { throw 'Compilation de l’interface interrompue.' }
} finally { Pop-Location }
& $python -m PyInstaller --noconfirm PrivexStudio.spec
if ($LASTEXITCODE -ne 0) { throw 'Compilation Windows interrompue.' }
Write-Host 'Application : dist/PrivexStudio/PrivexStudio.exe (conserver tout le dossier).'
