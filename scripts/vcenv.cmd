@echo off
rem Developer-environment wrapper: MSVC x64 + ninja + vcpkg, then run the given command.
rem Usage: scripts\vcenv.cmd <command> [args...]
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\vcpkg\downloads\tools\ninja-1.13.2-windows;%PATH%"
set "VCPKG_ROOT=C:\vcpkg"
%*
