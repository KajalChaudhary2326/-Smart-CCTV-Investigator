; ============================================================
;  Smart CCTV Investigator - Inno Setup Installer Script
; ============================================================
; HOW TO USE (do this on a Windows machine):
;   1. First run build_exe.bat (in the project root) - this creates
;      dist\SmartCCTVInvestigator\SmartCCTVInvestigator.exe
;   2. Download and install Inno Setup (free): https://jrsoftware.org/isdl.php
;   3. Open this file (SmartCCTVInvestigator.iss) in Inno Setup
;   4. Click Build > Compile (or press Ctrl+F9)
;   5. The final installer will appear in: installer\output\
;      as "SmartCCTVInvestigator_Setup_v1.0.exe"
;   6. That single Setup.exe is what you upload to GitHub Releases
;      and share with law enforcement / cyber cell users.
; ============================================================

#define MyAppName "Smart CCTV Investigator"
#define MyAppVersion "1.0"
#define MyAppPublisher "Your Organization Name"
#define MyAppExeName "SmartCCTVInvestigator.exe"

[Setup]
AppId={{B3F1E7A2-4C9D-4A6E-9B21-0C7F5D8E1A20}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; The exe built by PyInstaller (--onedir mode) lives here:
OutputDir=output
OutputBaseFilename=SmartCCTVInvestigator_Setup_v{#MyAppVersion}
SetupIconFile=..\assets\app_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Requires admin rights to install into Program Files (standard for desktop apps)
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; Pull in EVERYTHING PyInstaller produced (--onedir folder contents)
Source: "..\dist\SmartCCTVInvestigator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Bundle the user manual so it's available after install
Source: "..\USER_MANUAL.pdf"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\User Manual"; Filename: "{app}\USER_MANUAL.pdf"; Flags: createonlyiffileexists
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; Flags: nowait postinstall skipifsilent
