@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Build settings
set "BASE_PY=.venv\Scripts\python.exe"
set "TORCH_VERSION=2.14.1"
set "PYINSTALLER_VERSION=6.22.3"

rem Default: build both distributions
set "TARGET=%~1"
if not defined TARGET set "TARGET=all"

if /I "%TARGET%"=="cpu" goto target_ok
if /I "%TARGET%"=="cuda" goto target_ok
if /I "%TARGET%"=="all" goto target_ok

echo Usage: build.bat [cpu^|cuda^|all]
goto failed

:target_ok
if not exist "%BASE_PY%" (
    echo ERROR: Python was not found in .venv.
    echo Expected: %BASE_PY%
    goto failed
)

for %%F in (
    main.py
    app_version.py
    requirements.txt
    public_key.pem
    license.json
    UserGuide.html
    README.txt
) do (
    if not exist "%%F" (
        echo ERROR: Required file is missing: %%F
        goto failed
    )
)

if not exist "examples\" (
    echo ERROR: The examples directory is missing.
    goto failed
)

rem Read the application version from Python
set "APP_VERSION="
for /f "delims=" %%V in ('%BASE_PY% -c "from app_version import APP_VERSION; print(APP_VERSION)"') do set "APP_VERSION=%%V"

if not defined APP_VERSION (
    echo ERROR: Could not read APP_VERSION.
    goto failed
)

echo.
echo QNetLab version: %APP_VERSION%
echo Build target: %TARGET%
echo.

if /I "%TARGET%"=="cpu" goto cpu_only
if /I "%TARGET%"=="cuda" goto cuda_only

call :build cpu
if errorlevel 1 goto failed

call :build cuda
if errorlevel 1 goto failed

goto success

:cpu_only
call :build cpu
if errorlevel 1 goto failed
goto success

:cuda_only
call :build cuda
if errorlevel 1 goto failed
goto success


:build
set "FLAVOR=%~1"

if /I "%FLAVOR%"=="cpu" (
    set "ENV_DIR=.venv-build-cpu"
    set "WHEEL_FLAVOR=cpu"
    set "DIST_FLAVOR=cpu"
) else (
    set "ENV_DIR=.venv-build-cuda"
    set "WHEEL_FLAVOR=cu126"
    set "DIST_FLAVOR=cu126"
)

set "BUILD_PY=%ENV_DIR%\Scripts\python.exe"
set "DIST_DIR=dist\%APP_VERSION%-%DIST_FLAVOR%"

echo.
echo ==================================================
echo Building %FLAVOR%
echo ==================================================

if not exist "%BUILD_PY%" (
    echo Creating build environment...
    "%BASE_PY%" -m venv "%ENV_DIR%"
    if errorlevel 1 exit /b 1
)

echo Installing PyTorch %TORCH_VERSION%+%WHEEL_FLAVOR%...
"%BUILD_PY%" -m pip install ^
    "torch==%TORCH_VERSION%+%WHEEL_FLAVOR%" ^
    --index-url "https://download.pytorch.org/whl/%WHEEL_FLAVOR%"
if errorlevel 1 exit /b 1

echo Installing application and build dependencies...
"%BUILD_PY%" -m pip install ^
    -r requirements.txt ^
    "torch==%TORCH_VERSION%+%WHEEL_FLAVOR%" ^
    "pyinstaller==%PYINSTALLER_VERSION%" ^
    --extra-index-url "https://download.pytorch.org/whl/%WHEEL_FLAVOR%"
if errorlevel 1 exit /b 1

echo Checking dependencies...
"%BUILD_PY%" -m pip check
if errorlevel 1 exit /b 1

echo Checking PyTorch build...
"%BUILD_PY%" -c "import torch; expected='%TORCH_VERSION%+%WHEEL_FLAVOR%'; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); assert torch.__version__ == expected, 'Unexpected PyTorch build'"
if errorlevel 1 exit /b 1

echo Checking Python syntax...
"%BUILD_PY%" -c "import ast, pathlib; files=list(pathlib.Path('.').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p)) for p in files]; print('Syntax OK:', len(files), 'files')"
if errorlevel 1 exit /b 1

echo Building executable...
"%BUILD_PY%" -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --onedir ^
    --windowed ^
    --noupx ^
    --name QNetLab ^
    --distpath "%DIST_DIR%" ^
    --workpath "build\%FLAVOR%" ^
    --specpath "build\spec-%FLAVOR%" ^
    --add-data "public_key.pem:." ^
    --add-data "license.json:." ^
    --add-data "UserGuide.html:." ^
    --add-data "README.txt:." ^
    --add-data "examples:examples" ^
    main.py
if errorlevel 1 exit /b 1

if not exist "%DIST_DIR%\QNetLab\QNetLab.exe" (
    echo ERROR: Build finished but QNetLab.exe was not found.
    exit /b 1
)

echo Recording installed dependencies...
"%BUILD_PY%" -m pip freeze > "%DIST_DIR%\QNetLab\build-requirements.txt"
if errorlevel 1 exit /b 1

(
    echo Application: QNetLab
    echo Version: %APP_VERSION%
    echo Distribution: %DIST_FLAVOR%
    echo PyTorch: %TORCH_VERSION%+%WHEEL_FLAVOR%
    echo PyInstaller: %PYINSTALLER_VERSION%
) > "%DIST_DIR%\QNetLab\build-info.txt"

echo.
echo READY: %DIST_DIR%\QNetLab\QNetLab.exe
exit /b 0


:success
echo.
echo ==================================================
echo BUILD COMPLETED
echo Output directory: %CD%\dist
echo Test the executables before distribution.
echo ==================================================
pause
exit /b 0


:failed
echo.
echo ==================================================
echo BUILD FAILED
echo Read the error message above.
echo Do not distribute output from the failed build.
echo ==================================================
pause
exit /b 1