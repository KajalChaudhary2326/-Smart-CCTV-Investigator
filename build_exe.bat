@echo off
echo ============================================
echo  Smart CCTV Investigator - EXE Builder
echo ============================================
echo.
echo This will build a Windows application named
echo "Smart CCTV Investigator" with its custom icon.
echo.

pip install -r requirements.txt
pip install pyinstaller

REM --onedir   -> faster startup, files stay organized in one folder
REM              (better for installers than a single --onefile exe)
REM --windowed -> no console window pops up behind the GUI
REM --icon     -> uses our custom app icon everywhere in Windows
REM --name     -> this becomes the .exe name and folder name

pyinstaller --onedir --windowed ^
    --name "SmartCCTVInvestigator" ^
    --icon "assets\app_icon.ico" ^
    --add-data "assets;assets" ^
    main.py

echo.
echo ============================================
echo  Build complete.
echo  Find SmartCCTVInvestigator.exe inside:
echo  dist\SmartCCTVInvestigator\
echo ============================================
echo.
echo Next step: run installer\build_installer.bat
echo (or open installer\SmartCCTVInvestigator.iss in Inno Setup)
echo to create a proper Setup.exe installer.
echo.
pause
