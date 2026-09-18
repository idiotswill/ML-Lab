$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$standaloneExe = Join-Path $PWD "deployment\main.dist\MLLab.exe"
if (-not (Test-Path $standaloneExe)) {
    throw "Standalone build must exist before building the installer: $standaloneExe"
}

$version = (uv run python -c "from ml_lab import __version__; print(__version__)").Trim()
if (-not $version) {
    throw "Could not resolve ML Lab version"
}

$isccCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
$isccPath = if ($isccCommand) { $isccCommand.Source } else { $null }

if (-not $isccPath) {
    $candidates = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 7\ISCC.exe",
        "C:\Program Files\Inno Setup 7\ISCC.exe"
    )
    $isccPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $isccPath) {
    throw "ISCC.exe was not found; Windows packaging requires Inno Setup"
}

New-Item -ItemType Directory -Force "deployment\release" | Out-Null

& $isccPath "/DAppVersion=$version" "packaging\windows\MLLab.iss"
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compiler failed with exit code $LASTEXITCODE"
}

$installer = Join-Path $PWD "deployment\release\MLLab-Setup-$version-x64.exe"
if (-not (Test-Path $installer)) {
    throw "Installer compiler completed without producing $installer"
}

Write-Host "Developer installer build complete: $installer"
