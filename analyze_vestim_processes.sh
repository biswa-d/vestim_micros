#!/bin/bash
# Check VEstim Process Analysis

echo "🔍 VEstim Process Analysis"
echo "=========================="

# Count main launch_gui_qt processes
MAIN_PROCESSES=$(ps -u $(whoami) -o pid,ppid,command | grep "python -m launch_gui_qt" | grep -v grep | wc -l)
echo "📱 Main launch_gui_qt instances: $MAIN_PROCESSES"

# Count all launch_gui_qt related processes
ALL_PROCESSES=$(ps -u $(whoami) -o pid,ppid,command | grep launch_gui_qt | grep -v grep | wc -l)
echo "🔢 Total launch_gui_qt processes: $ALL_PROCESSES"

if [ $MAIN_PROCESSES -gt 0 ]; then
    echo ""
    echo "📊 Process breakdown:"
    echo "   - Main instances: $MAIN_PROCESSES"
    echo "   - Worker processes: $((ALL_PROCESSES - MAIN_PROCESSES))"
    echo "   - Workers per instance: $((($ALL_PROCESSES - $MAIN_PROCESSES) / $MAIN_PROCESSES))"
    
    echo ""
    echo "💾 Memory usage:"
    ps -u $(whoami) -o pid,rss,command | grep launch_gui_qt | grep -v grep | awk '{sum += $2} END {printf "   - Total memory: %.1f GB\n", sum/1024/1024}'
fi

echo ""
echo "🏃 Currently running main processes:"
ps -u $(whoami) -o pid,ppid,command | grep "python -m launch_gui_qt" | grep -v grep | head -10