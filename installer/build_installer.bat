@echo off
echo ============================================
echo  Smart CCTV Investigator - Installer Builder
echo ============================================
echo.
echo BEFORE running this, make sure you have:
echo  1. Run build_exe.bat already (creates dist\SmartCCTVInvestigator\)
echo  2. Installed Inno Setup from https://jrsoftware.org/isdl.php
echo.
echo This script will try to compile the installer using Inno Setup's
echo command-line compiler (ISCC.exe). If Inno Setup is installed in
echo the default location, this should work automatically.
echo.
pause

set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if not exist %ISCC% (
    echo.
    echo Could not find ISCC.exe at the default location.
    echo Please open installer\SmartCCTVInvestigator.iss manually in the
    echo Inno Setup app and click Build - Compile instead.
    echo.
    pause
    exit /b 1
)

%ISCC% "SmartCCTVInvestigator.iss"

echo.
echo ============================================
echo  Done. Check installer\output\ for:
echo  SmartCCTVInvestigator_Setup_v1.0.exe
echo  This is the file you share with users.
echo ============================================
pause
