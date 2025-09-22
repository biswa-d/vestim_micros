#!/bin/bash
# VEstim Process Cleanup Script
# Use this to manually clean up any remaining VEstim processes

echo "🔍 Searching for VEstim-related processes..."

# Find all processes related to launch_gui_qt
VESTIM_PROCESSES=$(ps -u $(whoami) -o pid,command | grep -i launch_gui_qt | grep -v grep)

if [ -z "$VESTIM_PROCESSES" ]; then
    echo "✅ No VEstim processes found."
    exit 0
fi

echo "📋 Found VEstim processes:"
echo "$VESTIM_PROCESSES"
echo ""

# Count processes
PROCESS_COUNT=$(echo "$VESTIM_PROCESSES" | wc -l)
echo "📊 Total processes found: $PROCESS_COUNT"

# Ask for confirmation
read -p "❓ Do you want to terminate all these processes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 Terminating VEstim processes..."
    
    # First try graceful termination
    echo "$VESTIM_PROCESSES" | awk '{print $1}' | xargs -r kill -TERM
    
    # Wait a moment
    sleep 2
    
    # Check what's still running
    REMAINING=$(ps -u $(whoami) -o pid,command | grep -i launch_gui_qt | grep -v grep)
    
    if [ ! -z "$REMAINING" ]; then
        echo "⚠️  Some processes still running, force killing..."
        echo "$REMAINING" | awk '{print $1}' | xargs -r kill -KILL
        sleep 1
    fi
    
    # Final check
    FINAL_CHECK=$(ps -u $(whoami) -o pid,command | grep -i launch_gui_qt | grep -v grep)
    
    if [ -z "$FINAL_CHECK" ]; then
        echo "✅ All VEstim processes terminated successfully!"
    else
        echo "❌ Some processes may still be running:"
        echo "$FINAL_CHECK"
    fi
else
    echo "❌ Operation cancelled."
fi

echo ""
echo "💡 To prevent this in the future:"
echo "   - Use Ctrl+C to stop the application"
echo "   - Close GUI windows properly"
echo "   - Check that training has completed before closing"