@echo off
rem ============================================================
rem  SGR A* Scientific Simulator - one-click launcher
rem  Double-click to start. Boots to the cinematic main menu.
rem
rem  - Makes the Vulkan validation layers findable (VK_LAYER_PATH)
rem  - Builds the engine first if it has never been built
rem  - Any arguments are passed straight to the engine, e.g.:
rem      LAUNCH.bat --validation          (turn validation layers ON)
rem      LAUNCH.bat --preset 3            (start with the ARCTIC preset)
rem      LAUNCH.bat --shader disk --hud   (skip the menu)
rem ============================================================
setlocal
pushd "%~dp0"

rem Vulkan validation layers (vcpkg build) - available on demand
set "VK_LAYER_PATH=C:\vcpkg\installed\x64-windows\bin"

set "ENGINE=build\engine-rel\engine\sgr_engine.exe"

if not exist "%ENGINE%" (
    echo [launcher] Engine not built yet - building engine-rel now.
    echo [launcher] The first build lets vcpkg compile dependencies and
    echo [launcher] can take a long time. Subsequent launches are instant.
    call scripts\vcenv.cmd cmake --preset engine-rel || goto :fail
    call scripts\vcenv.cmd cmake --build --preset engine-rel || goto :fail
)

if not exist "%ENGINE%" goto :fail

echo [launcher] Starting SGR A* simulator...
"%ENGINE%" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo [launcher] Engine exited with code %RC%.
    pause
)
popd
exit /b %RC%

:fail
echo.
echo [launcher] Build failed or engine missing. See output above.
echo [launcher] You can build manually with:
echo     scripts\vcenv.cmd cmake --preset engine-rel
echo     scripts\vcenv.cmd cmake --build --preset engine-rel
pause
popd
exit /b 1
