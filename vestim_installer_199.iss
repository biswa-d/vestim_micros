; Inno Setup Script for Vestim (199 Env)
; This creates a professional Windows installer

#define MyAppName "Vestim"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "B Dehury (battery.mcmaster.ca)"
#define MyAppURL "https://github.com/biswa-d/vestim_micros"
#define MyAppExeName "Vestim_2.0.0_2025_August_21_tvo_199_pack.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{8A8F6C8B-7B5C-4D8E-9F2A-1E3B4C5D6E7F}}
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
#define BuildDate GetDateTimeString('yyyy-mm-dd', '-', ':')
OutputBaseFilename=vestim-installer-{#MyAppVersion}-{#BuildDate}-199
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
Name: "vc2019redist"; Description: "Install Microsoft Visual C++ 2015-2022 Redistributable"; GroupDescription: "Required System Components"; Flags: checkedonce
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "vestim\gui\resources\*"; DestDir: "{app}\resources"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "packaging\USER_README.md"; DestDir: "{app}"; DestName: "USER_README.md"; Flags: ignoreversion
Source: "packaging\MODEL_DEPLOYMENT_GUIDE.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "hyperparams.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "defaults_templates\*"; DestDir: "{app}\defaults_templates"; Flags: ignoreversion recursesubdirs createallsubdirs
; Demo data files and hyperparams templates will be copied to projects folder by installer script

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Microsoft Visual C++ 2015-2022 Redistributable..."; Tasks: vc2019redist; Check: VCRedistNeedsInstallAndDownload
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

function VCRedistNeedsInstall: Boolean;
var
    Version: String;
begin
    Result := not RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Version', Version);
end;

function DownloadVCRedist(): Boolean;
var
  WinHttp: Variant;
  Stream: Variant;
  URL, FileName: String;
begin
  Result := False;
  URL := 'https://aka.ms/vs/17/release/vc_redist.x64.exe';
  FileName := ExpandConstant('{tmp}\vc_redist.x64.exe');
  
  try
    WinHttp := CreateOleObject('WinHttp.WinHttpRequest.5.1');
    WinHttp.Open('GET', URL, False);
    WinHttp.Send;

    if WinHttp.Status = 200 then
    begin
      Stream := CreateOleObject('Adodb.Stream');
      Stream.Open;
      Stream.Write(WinHttp.ResponseBody);
      Stream.SaveToFile(FileName, 2);
      Stream.Close;
      Result := True;
    end else
    begin
      MsgBox('Failed to download a required component (Visual C++ Redistributable). Please check your internet connection and try again.', mbError, MB_OK);
    end;
  except
    MsgBox('An error occurred while downloading a required component (Visual C++ Redistributable). Please check your internet connection and try again.', mbError, MB_OK);
  end;
end;

function VCRedistNeedsInstallAndDownload: Boolean;
begin
  if VCRedistNeedsInstall() then
  begin
    Result := DownloadVCRedist();
  end
  else
  begin
    Result := False;
  end;
end;

procedure CopyFolder(Source, Dest: String);
var
  FindRec: TFindRec;
  SourcePath, DestPath: String;
begin
  if FindFirst(Source + '\*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          SourcePath := Source + '\' + FindRec.Name;
          DestPath := Dest + '\' + FindRec.Name;
          if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY = 0 then
          begin
            if not DirExists(Dest) then
              ForceDirectories(Dest);
            FileCopy(SourcePath, DestPath, False);
          end
          else
          begin
            CopyFolder(SourcePath, DestPath);
          end;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

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
  DataPath: String;
  HyperparamsPath: String;
  ConfigFile: String;
  SettingsFile: String;
  ConfigContent: String;
  SettingsContent: String;
  EscapedPath: String;
  EscapedDataPath: String;
  I: Integer;
  AppDir: String;
begin
  // Handle the existing upgrade logic
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  end;
  
  // Create projects folder structure and copy demo files after installation
  if CurStep = ssPostInstall then
  begin
    // Get paths
    AppDir := ExpandConstant('{app}');
    ProjectsBasePath := ProjectsFolderPage.Values[0];
    ProjectsFullPath := ProjectsBasePath + '\vestim_projects';
    DataPath := ProjectsFullPath + '\data';
    HyperparamsPath := ProjectsFullPath + '\default_hyperparams';
    
    // Create the directory structure
    if not DirExists(ProjectsFullPath) then
      ForceDirectories(ProjectsFullPath);
    if not DirExists(DataPath) then
      ForceDirectories(DataPath);
    if not DirExists(DataPath + '\train_data') then
      ForceDirectories(DataPath + '\train_data');
    if not DirExists(DataPath + '\val_data') then
      ForceDirectories(DataPath + '\val_data');
    if not DirExists(DataPath + '\test_data') then
      ForceDirectories(DataPath + '\test_data');
    if not DirExists(HyperparamsPath) then
      ForceDirectories(HyperparamsPath);
    
    // Copy demo data files from the installed data folder to the user's project folder
    // Recursively copy demo data from the installed data folder to the user's project folder
    if DirExists(AppDir + '\data\train_data') then
      CopyFolder(AppDir + '\data\train_data', DataPath + '\train_data');
    if DirExists(AppDir + '\data\val_data') then
      CopyFolder(AppDir + '\data\val_data', DataPath + '\val_data');
    if DirExists(AppDir + '\data\test_data') then
      CopyFolder(AppDir + '\data\test_data', DataPath + '\test_data');
    
    // Copy hyperparams template files from the installed defaults_templates folder to the user's project folder
    if DirExists(AppDir + '\defaults_templates') then
      CopyFolder(AppDir + '\defaults_templates', HyperparamsPath);
    
    // Escape backslashes for JSON - manual replacement for projects path
    EscapedPath := '';
    for I := 1 to Length(ProjectsFullPath) do
    begin
      if ProjectsFullPath[I] = '\' then
        EscapedPath := EscapedPath + '\\'
      else
        EscapedPath := EscapedPath + ProjectsFullPath[I];
    end;
    
    // Escape backslashes for JSON - manual replacement for data path
    EscapedDataPath := '';
    for I := 1 to Length(DataPath) do
    begin
      if DataPath[I] = '\' then
        EscapedDataPath := EscapedDataPath + '\\'
      else
        EscapedDataPath := EscapedDataPath + DataPath[I];
    end;
    
    // Build JSON config content for vestim_config.json
    ConfigContent := '{';
    ConfigContent := ConfigContent + #13#10;
    ConfigContent := ConfigContent + '  "projects_directory": "' + EscapedPath + '",';
    ConfigContent := ConfigContent + #13#10;
    ConfigContent := ConfigContent + '  "data_directory": "' + EscapedDataPath + '",';
    ConfigContent := ConfigContent + #13#10;
    ConfigContent := ConfigContent + '  "created_by_installer": true';
    ConfigContent := ConfigContent + #13#10;
    ConfigContent := ConfigContent + '}';
    
    // Save the config file in the application directory
    ConfigFile := AppDir + '\vestim_config.json';
    SaveStringToFile(ConfigFile, ConfigContent, False);
    
    // Build JSON content for default_settings.json
    SettingsContent := '{';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '  "last_used": {';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '    "train_folder": "' + EscapedDataPath + '\\train_data",';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '    "val_folder": "' + EscapedDataPath + '\\val_data",';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '    "test_folder": "' + EscapedDataPath + '\\test_data",';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '    "file_format": "csv"';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '  },';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '  "default_folders": {';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '    "train_folder": "' + EscapedDataPath + '\\train_data",';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '    "val_folder": "' + EscapedDataPath + '\\val_data",';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '    "test_folder": "' + EscapedDataPath + '\\test_data"';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '  }';
    SettingsContent := SettingsContent + #13#10;
    SettingsContent := SettingsContent + '}';
    
    // Save the default settings file in the projects directory
    SettingsFile := ProjectsFullPath + '\default_settings.json';
    SaveStringToFile(SettingsFile, SettingsContent, False);
    
    // Copy the comprehensive user README to projects directory
    if FileExists(AppDir + '\USER_README.md') then
      FileCopy(AppDir + '\USER_README.md', ProjectsFullPath + '\README.md', False)
    else if FileExists(AppDir + '\README.md') then
      FileCopy(AppDir + '\README.md', ProjectsFullPath + '\README.md', False);
    
    // Copy the Model Deployment Guide to projects directory
    if FileExists(AppDir + '\MODEL_DEPLOYMENT_GUIDE.md') then
      FileCopy(AppDir + '\MODEL_DEPLOYMENT_GUIDE.md', ProjectsFullPath + '\MODEL_DEPLOYMENT_GUIDE.md', False);
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