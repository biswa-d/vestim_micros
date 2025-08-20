#!/usr/bin/env python3
"""
Test script to verify timer functionality in the training GUI.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, '/mnt/data1/dehuryb/vestim-gpu-2/VS_Jobs/vestim_micros')

def test_timer_formatting():
    """Test the timer formatting functions."""
    from vestim.gateway.src.training_task_manager_qt import format_time, format_time_long
    
    # Test format_time (mm:ss)
    print("Testing format_time (mm:ss):")
    test_cases = [0, 30, 60, 90, 120, 150, 3600]
    for seconds in test_cases:
        formatted = format_time(seconds)
        print(f"  {seconds} seconds -> {formatted}")
    
    print("\nTesting format_time_long (hh:mm:ss):")
    test_cases = [0, 30, 60, 90, 120, 150, 3600, 3660, 7200]
    for seconds in test_cases:
        formatted = format_time_long(seconds)
        print(f"  {seconds} seconds -> {formatted}")

def test_timer_logic():
    """Test the timer logic simulation."""
    print("\nTesting timer logic simulation:")
    
    # Simulate job start time
    job_start_time = time.time()
    print(f"Job started at: {job_start_time}")
    
    # Simulate task 1
    task1_start_time = time.time()
    print(f"Task 1 started at: {task1_start_time}")
    
    # Simulate some progress
    time.sleep(0.1)  # Small delay to simulate progress
    
    # Calculate timers
    task1_elapsed = time.time() - task1_start_time
    job_elapsed = time.time() - job_start_time
    
    print(f"Task 1 elapsed: {task1_elapsed:.2f}s")
    print(f"Job elapsed: {job_elapsed:.2f}s")
    
    # Simulate task 2 (new task, same job)
    task2_start_time = time.time()
    print(f"Task 2 started at: {task2_start_time}")
    
    # Simulate some progress
    time.sleep(0.1)
    
    # Calculate timers
    task2_elapsed = time.time() - task2_start_time
    job_elapsed = time.time() - job_start_time
    
    print(f"Task 2 elapsed: {task2_elapsed:.2f}s")
    print(f"Job elapsed: {job_elapsed:.2f}s")
    
    print("\nTimer logic test completed successfully!")

if __name__ == "__main__":
    print("Testing timer functionality...")
    test_timer_formatting()
    test_timer_logic()
    print("\nAll tests completed!")
