@echo off
setlocal

set "MLLAB_EXE=%LOCALAPPDATA%\Programs\Frankenhomie ML Lab\MLLab.exe"
set "RECEIPT=%~dp0MLLab-v0.1.0-testing.3-representative-performance.json"

if not exist "%MLLAB_EXE%" (
    echo Frankenhomie ML Lab is not installed at the default per-user location:
    echo   %MLLAB_EXE%
    echo Install MLLab-Setup-v0.1.0-testing.3-x64.exe first, then run this file again.
    pause
    exit /b 2
)

if exist "%RECEIPT%" del /q "%RECEIPT%"

echo Running the full representative-machine Phase 4 performance gate.
echo This writes evidence only. It does not integrate anything into Frankenhomie.
"%MLLAB_EXE%" --release-performance-evidence "%RECEIPT%"
set "RESULT=%ERRORLEVEL%"

if not exist "%RECEIPT%" (
    echo.
    echo FAILED: the application did not produce a performance receipt.
    pause
    exit /b 3
)

echo.
if "%RESULT%"=="0" (
    echo PASS: representative-machine performance gate passed.
    echo Receipt:
    echo   %RECEIPT%
) else (
    echo FAIL: one or more representative-machine performance gates did not pass.
    echo The receipt was preserved for diagnosis:
    echo   %RECEIPT%
)
echo.
echo INTEGRATION_GATE remains NO_GO.
pause
exit /b %RESULT%
