#!/bin/bash
# Quick Packaging Verification Script
# Checks if everything is ready for building Vestim

echo "🔍 VEstim Packaging Verification"
echo "================================"

# Check current git status
echo "📋 Git Status:"
echo "Branch: $(git branch --show-current)"
echo "Last commit: $(git log --oneline -1)"
echo "Clean repo: $(git status --porcelain | wc -l) uncommitted changes"
echo ""

# Check Python environment
echo "🐍 Python Environment:"
python --version
pip list | grep -E "(pyinstaller|PyQt5|torch|pandas|numpy|matplotlib)" | head -10
echo ""

# Check key files exist
echo "📁 Key Files Check:"
files=(
    "launch_gui_qt.py"
    "vestim_installer.iss" 
    "packaging/build_exe.py"
    "packaging/MODEL_DEPLOYMENT_GUIDE.md"
    "USER_README.md"
    "hyperparams.json"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
    fi
done
echo ""

# Expected output
echo "📦 Expected Build Output:"
branch=$(git branch --show-current)
date=$(date +"%Y_%B_%d")
echo "Executable: Vestim_2.0.1_${date}_${branch}.exe"
echo "Console window: ✅ Enabled (logs visible)"
echo "Includes: GUI, Terminal, Model Deployment Guide"
echo ""

echo "🚀 Ready to build!"
echo "Run: python packaging/build_exe.py"