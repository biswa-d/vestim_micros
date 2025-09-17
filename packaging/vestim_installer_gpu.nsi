# NSIS Script for Vestim Installer (GPU General)

!define APP_NAME "Vestim"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "Biswanath Dehury"
!define APP_URL "https://github.com/yourusername/vestim"
!define APP_EXE "Vestim.exe"
!define GPU_SETUP_SCRIPT "gpu_setup.py"

# Installer settings
Name "${APP_NAME}"
OutFile "installer_output\vestim-installer-${APP_VERSION}-gpu.exe"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
RequestExecutionLevel user

# Modern UI
!include "MUI2.nsh"

# UI Settings
!define MUI_ABORTWARNING
!define MUI_ICON "vestim\gui\resources\icon.ico"
!define MUI_UNICON "vestim\gui\resources\icon.ico"

# Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

# Languages
!insertmacro MUI_LANGUAGE "English"

# Installer sections
Section "Main Application" SecMain
    SetOutPath "$INSTDIR"
    File "dist\${APP_EXE}"
    File "dist\${GPU_SETUP_SCRIPT}"
    File "README.md"
    File "packaging\USER_README.md"
    File "packaging\MODEL_DEPLOYMENT_GUIDE.md"
    File "LICENSE"
    File "hyperparams.json"
    
    SetOutPath "$INSTDIR\resources"
    File /r "vestim\gui\resources\*"
    
    # Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    # Registry entries
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "URLInfoAbout" "${APP_URL}"
SectionEnd

Section "Install GPU Support" SecGpu
    DetailPrint "Installing GPU support for PyTorch..."
    ExecWait '"$INSTDIR\${APP_EXE}" "${GPU_SETUP_SCRIPT}"'
SectionEnd

Section "Desktop Shortcut" SecDesktop
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
SectionEnd

Section "Start Menu" SecStartMenu
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

# Uninstaller
Section "Uninstall"
    Delete "$INSTDIR\${APP_EXE}"
    Delete "$INSTDIR\${GPU_SETUP_SCRIPT}"
    Delete "$INSTDIR\README.md"
    Delete "$INSTDIR\USER_README.md"
    Delete "$INSTDIR\MODEL_DEPLOYMENT_GUIDE.md"
    Delete "$INSTDIR\LICENSE"
    Delete "$INSTDIR\hyperparams.json"
    Delete "$INSTDIR\Uninstall.exe"
    
    RMDir /r "$INSTDIR\resources"
    RMDir "$INSTDIR"
    
    Delete "$DESKTOP\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
SectionEnd