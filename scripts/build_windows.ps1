$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

uv sync --extra dev

# Use Nuitka directly instead of pyside6-deploy so CI owns the exact compiler
# contract. In particular, --assume-yes-for-downloads permits Nuitka's
# dependency-analysis helper to be fetched non-interactively on clean runners.
$nuitkaArgs = @(
    "-m", "nuitka",
    "main.py",
    "--mode=standalone",
    "--enable-plugin=pyside6",
    "--assume-yes-for-downloads",
    "--output-dir=deployment",
    "--output-filename=MLLab.exe",
    "--include-data-dir=src/ml_lab/ui/qml=ml_lab/ui/qml",
    "--noinclude-qt-translations"
)
uv run python @nuitkaArgs

$exe = Get-ChildItem -Path "deployment" -Recurse -Filter "MLLab.exe" | Select-Object -First 1
if (-not $exe) {
    throw "Standalone build command completed without producing MLLab.exe"
}

Write-Host "Development standalone build complete: $($exe.FullName) ($($exe.Length) bytes)"
Write-Host "Phase 4 owns installer/productization."
