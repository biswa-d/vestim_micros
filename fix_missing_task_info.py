#!/usr/bin/env python3
"""
Utility script to fix missing task_info.json files in existing job folders.
This creates individual task_info.json files in each task directory based on the 
architecture-level task_info.json and training_tasks_summary.json files.
"""

import os
import json
import glob

def fix_missing_task_info_files(job_folder):
    """Fix missing task_info.json files in task directories."""
    models_dir = os.path.join(job_folder, 'models')
    
    if not os.path.exists(models_dir):
        print(f"No models directory found in {job_folder}")
        return
    
    print(f"Fixing task_info.json files in {job_folder}")
    
    # Check for training_tasks_summary.json
    tasks_summary_file = os.path.join(job_folder, 'training_tasks_summary.json')
    tasks_from_summary = []
    
    if os.path.exists(tasks_summary_file):
        with open(tasks_summary_file, 'r') as f:
            tasks_from_summary = json.load(f)
        print(f"Found {len(tasks_from_summary)} tasks in training_tasks_summary.json")
    
    fixed_count = 0
    
    # Process each architecture directory
    for arch_folder in os.listdir(models_dir):
        arch_path = os.path.join(models_dir, arch_folder)
        if not os.path.isdir(arch_path):
            continue
            
        # Check for architecture-level task_info.json
        arch_task_info = os.path.join(arch_path, 'task_info.json')
        base_task_info = None
        
        if os.path.exists(arch_task_info):
            with open(arch_task_info, 'r') as f:
                base_task_info = json.load(f)
        
        # Process each task directory
        for task_folder in os.listdir(arch_path):
            task_path = os.path.join(arch_path, task_folder)
            if not os.path.isdir(task_path):
                continue
                
            task_info_file = os.path.join(task_path, 'task_info.json')
            
            if os.path.exists(task_info_file):
                print(f"  ✓ {arch_folder}/{task_folder} already has task_info.json")
                continue
            
            # Find matching task info from summary or use base info
            task_info_to_save = None
            
            # First try to find exact match in tasks summary
            for task in tasks_from_summary:
                if task.get('task_dir', '').endswith(task_folder):
                    task_info_to_save = task.copy()
                    break
            
            # If not found in summary, create from base_task_info
            if not task_info_to_save and base_task_info:
                task_info_to_save = base_task_info.copy()
                # Update paths for this specific task
                task_info_to_save['task_dir'] = task_path
                task_info_to_save['task_id'] = f"task_{task_folder}"
                
                # Try to extract repetition number from folder name
                if '_rep_' in task_folder:
                    try:
                        rep_part = task_folder.split('_rep_')[-1]
                        rep_num = int(rep_part)
                        task_info_to_save['repetition'] = rep_num
                        if 'hyperparams' in task_info_to_save:
                            task_info_to_save['hyperparams']['CURRENT_REPETITION'] = rep_num
                    except:
                        pass
            
            if task_info_to_save:
                # Save the task_info.json in the task directory
                with open(task_info_file, 'w') as f:
                    json.dump(task_info_to_save, f, indent=4)
                print(f"  ✓ Created {arch_folder}/{task_folder}/task_info.json")
                fixed_count += 1
            else:
                print(f"  ✗ Could not create task_info.json for {arch_folder}/{task_folder}")
    
    print(f"\nFixed {fixed_count} missing task_info.json files")

if __name__ == "__main__":
    # Fix the current job folder
    job_folder = r"c:\Users\dehuryb\code\vestim_micros\output\job_20250904-123450"
    fix_missing_task_info_files(job_folder)
    
    print("\n🎉 Task info files have been fixed!")
    print("You can now run the standalone testing again.")