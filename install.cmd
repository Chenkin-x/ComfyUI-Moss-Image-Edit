@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "MOSS_PYTHON="
set "MOSS_COUNT=0"
for %%I in ("%~dp0..\..") do set "MOSS_COMFYUI=%%~fI"
if not "%~1"=="" (
    set "MOSS_PYTHON=%~1"
    goto run
)
for %%P in ("%MOSS_COMFYUI%\.venv\Scripts\python.exe" "%MOSS_COMFYUI%\venv\Scripts\python.exe" "%MOSS_COMFYUI%\..\python_embeded\python.exe" "%MOSS_COMFYUI%\..\python\python.exe") do (
    if exist "%%~fP" (
        echo Found Python: %%~fP
        set "MOSS_PYTHON=%%~fP"
        set /a MOSS_COUNT+=1 >nul
    )
)
if not "%MOSS_COUNT%"=="1" (
    echo [MOSS Edit] 无法唯一确定 ComfyUI 的 Python，不使用系统 Python 猜测。
    echo Cannot identify one ComfyUI Python. Run with the exact interpreter:
    echo install.cmd "X:\ComfyUI-folder\python\python.exe"
    pause
    exit /b 3
)
:run
if not exist "%MOSS_PYTHON%" (
    echo [FAIL] Python not found: %MOSS_PYTHON%
    pause
    exit /b 3
)
echo 请核对下面的路径是否为启动器实际使用的 Python。
echo Verify that this is the Python used by your ComfyUI launcher.
"%MOSS_PYTHON%" -X utf8 -B "%~dp0install.py" --comfyui "%MOSS_COMFYUI%"
set "MOSS_RESULT=%ERRORLEVEL%"
echo.
echo Exit code: %MOSS_RESULT% ^(0=passed, 2=insufficient, 3=check incomplete^)
pause
exit /b %MOSS_RESULT%
