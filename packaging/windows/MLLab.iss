#ifndef AppVersion
  #define AppVersion "0.1.0.dev0"
#endif

#define AppName "Frankenhomie ML Lab"
#define AppExeName "MLLab.exe"

[Setup]
AppId={{7D9C7F4B-7D0D-4FA6-9F3B-7D819A2DFA54}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Frankenhomie ML Lab
DefaultDirName={localappdata}\Programs\Frankenhomie ML Lab
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
OutputDir=..\..\deployment\release
OutputBaseFilename=MLLab-Setup-v{#AppVersion}-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

[Files]
Source: "..\..\deployment\main.dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
