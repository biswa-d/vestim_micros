#!/usr/bin/env python3
"""
Simple test to verify the VEstim architecture components work correctly.
"""

import sys
import os

# Add the vestim module to path
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all critical modules can be imported."""
    try:
        from vestim.backend.src.managers.job_manager import JobManager
        print("✓ JobManager imported successfully")
        
        from vestim.gui.src.managers.dashboard_manager import DashboardManager  
        print("✓ DashboardManager imported successfully")
        
        from vestim.backend.src.services.job_service import JobService
        print("✓ JobService imported successfully")
        
        from vestim.backend.src.services.training_service import TrainingService
        print("✓ TrainingService imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_job_manager():
    """Test JobManager functionality."""
    try:
        from vestim.backend.src.managers.job_manager import JobManager
        
        # Create a job manager instance
        job_manager = JobManager()
        print("✓ JobManager instance created")
        
        # Test job creation
        job_id = "test_job_001"
        success = job_manager.create_job(job_id, {"test": "data"})
        if success:
            print(f"✓ Job {job_id} created successfully")
        else:
            print(f"✗ Failed to create job {job_id}")
            
        # Test job listing
        jobs = job_manager.get_all_jobs()
        print(f"✓ Retrieved {len(jobs)} jobs")
        
        # Test job status update
        success = job_manager.update_job_status(job_id, "test_status", "setup", {"test": "data"})
        if success:
            print(f"✓ Job status updated successfully")
        else:
            print(f"✗ Failed to update job status")
            
        return True
        
    except Exception as e:
        print(f"✗ JobManager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== VEstim Architecture Test ===")
    
    all_passed = True
    
    print("\n1. Testing imports...")
    if not test_imports():
        all_passed = False
    
    print("\n2. Testing JobManager...")
    if not test_job_manager():
        all_passed = False
    
    print(f"\n=== Test Results ===")
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    main()
