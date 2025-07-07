; Inno Setup Script for Vestim
; This creates a professional Windows installer

#define MyAppName "Vestim"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Biswanath Dehury"
#define MyAppURL "https://github.com/yourusername/vestim"
#define MyAppExeName "Vestim.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{8A8F6C8B-7B5C-4D8E-9F2A-1E3B4C5D6E7F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
InfoBeforeFile=INSTALL_INFO.txt
OutputDir=installer_output
OutputBaseFilename=vestim-installer-{#MyAppVersion}
SetupIconFile=vestim\gui\resources\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
Source: "dist\Vestim.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "vestim\gui\resources\*"; DestDir: "{app}\resources"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "hyperparams.json"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[CustomMessages]
ProjectsFolderPage_Caption=Select Projects Location
ProjectsFolderPage_Description=Choose where Vestim will store your project files
ProjectsFolderPage_SubCaption=Vestim will create a "vestim_projects" folder in the location you select.

[Code]
var
  ProjectsFolderPage: TInputDirWizardPage;

function GetUninstallString(): String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade(): Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion(): Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
  Result := 0;
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES','', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end else
    Result := 1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ProjectsBasePath: String;
  ProjectsFullPath: String;
  ConfigFile: String;
  ConfigContent: String;
  EscapedPath: String;
  I: Integer;
begin
  // Handle the existing upgrade logic
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  end;
  
  // Create projects folder and save config after installation
  if CurStep = ssPostInstall then
  begin
    // Get the selected base path and create vestim_projects folder
    ProjectsBasePath := ProjectsFolderPage.Values[0];
    ProjectsFullPath := ProjectsBasePath + '\vestim_projects';
    
    // Create the vestim_projects directory
    if not DirExists(ProjectsFullPath) then
      ForceDirectories(ProjectsFullPath);
    
    // Escape backslashes for JSON - manual replacement
    EscapedPath := '';
    for I := 1 to Length(ProjectsFullPath) do
    begin
      if ProjectsFullPath[I] = '\' then
        EscapedPath := EscapedPath + '\\'
      else
        EscapedPath := EscapedPath + ProjectsFullPath[I];
    end;
    
    // Build JSON config content step by step
    ConfigContent := '{';
    ConfigContent := ConfigContent + #13#10;
    ConfigContent := ConfigContent + '  "projects_directory": "' + EscapedPath + '",';
    ConfigContent := ConfigContent + #13#10;
    ConfigContent := ConfigContent + '  "created_by_installer": true';
    ConfigContent := ConfigContent + #13#10;
    ConfigContent := ConfigContent + '}';
    
    // Save the config file
    ConfigFile := ExpandConstant('{app}\vestim_config.json');
    SaveStringToFile(ConfigFile, ConfigContent, False);
  end;
end;

procedure InitializeWizard;
begin
  // Create the projects folder selection page (after installation directory)
  ProjectsFolderPage := CreateInputDirPage(wpSelectDir,
    CustomMessage('ProjectsFolderPage_Caption'),
    CustomMessage('ProjectsFolderPage_Description'), 
    CustomMessage('ProjectsFolderPage_SubCaption'),
    False, '');
  
  // Add input field with default path (user's Documents folder)
  ProjectsFolderPage.Add('&Location for vestim_projects folder:');
  ProjectsFolderPage.Values[0] := ExpandConstant('{userdocs}');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  
  // Validate the projects folder path
  if CurPageID = ProjectsFolderPage.ID then
  begin
    if ProjectsFolderPage.Values[0] = '' then
    begin
      MsgBox('Please select a valid location for the vestim_projects folder.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;
