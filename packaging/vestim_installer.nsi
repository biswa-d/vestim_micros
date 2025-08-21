# NSIS Script for Vestim Installer
# Alternative to Inno Setup

!define APP_NAME "Vestim"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "Biswanath Dehury"
!define APP_URL "https://github.com/yourusername/vestim"
!define APP_EXE "Vestim.exe"

# Installer settings
Name "${APP_NAME}"
OutFile "installer_output\vestim-installer-${APP_VERSION}.exe"
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

Page custom GpuSelectionPage
!define MUI_PAGE_CUSTOMFUNCTION_PRE GpuSelectionPagePre

!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

# Languages
!insertmacro MUI_LANGUAGE "English"

# Installer sections
Function GpuSelectionPage
    !insertmacro MUI_HEADER_TEXT "GPU Selection" "Choose whether to install GPU support."
    
    nsDialogs::Create 1018
    Pop $0

    ${NSD_CreateLabel} 0 0 100% 12u "Do you want to install GPU support for PyTorch?"
    Pop $0
    
    ${NSD_CreateRadioButton} 10u 20u 80% 8u "Yes, install GPU support"
    Pop $hRadioYes
    
    ${NSD_CreateRadioButton} 10u 35u 80% 8u "No, continue with CPU-only version"
    Pop $hRadioNo
    
    ${If} $GpuSupport == "1"
        ${NSD_Check} $hRadioYes
    ${Else}
        ${NSD_Check} $hRadioNo
    ${EndIf}
    
    nsDialogs::Show
FunctionEnd

Function GpuSelectionPagePre
    ${If} $GpuSupport == ""
        StrCpy $GpuSupport "0"
    ${EndIf}
FunctionEnd

Section "Main Application" SecMain
    SetOutPath "$INSTDIR"
    File "dist\${APP_EXE}"
    File "README.md"
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

Section "GPU Support" SecGpu
    SectionIn RO
    
    DetailPrint "Installing GPU support for PyTorch..."
    
    ExecWait '"$INSTDIR\${APP_EXE}" --install-gpu'
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
    Delete "$INSTDIR\README.md"
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
