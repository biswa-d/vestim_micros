#!/usr/bin/env python3
"""
Test script to verify JobContainer-based migration is working correctly.
This script tests the core functionality of the refactored system.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_job_container_basic():
    """Test basic JobContainer functionality"""
    print("Testing JobContainer basic functionality...")
    
    from vestim.backend.src.managers.job_container import JobContainer
    
    # Create a temporary job folder
    temp_dir = tempfile.mkdtemp()
    try:
        job_id = "test_job_001"
        selections = {"model_type": "test", "data_path": "/tmp/test"}
        
        # Create job container
        container = JobContainer(job_id, temp_dir, selections)
        
        # Test basic properties
        assert container.job_id == job_id
        assert container.job_folder == temp_dir
        assert container.selections == selections
        assert container.status == "created"
        
        # Test status updates
        container.update_status("running", "Job is running", 25.0)
        assert container.status == "running"
        assert container.progress_percent == 25.0
        
        # Test task progress
        container.update_task_progress("task_1", {"status": "training", "epoch": 5})
        assert "task_1" in container.task_progress
        
        # Test manager creation
        training_manager = container.get_training_task_manager()
        assert training_manager is not None
        assert "training_task_manager" in container.managers
        
        print("✓ JobContainer basic functionality test passed")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_job_manager_migration():
    """Test JobManager with new JobContainer architecture"""
    print("Testing JobManager migration...")
    
    from vestim.backend.src.managers.job_manager import JobManager
    
    # Create a temporary output directory
    temp_dir = tempfile.mkdtemp()
    original_output_dir = None
    
    try:
        # Temporarily override OUTPUT_DIR
        import vestim.config as config
        original_output_dir = config.OUTPUT_DIR
        config.OUTPUT_DIR = temp_dir
        
        # Create JobManager instance
        job_manager = JobManager()
        
        # Test job creation
        selections = {"model_type": "test", "data_path": "/tmp/test"}
        job_id = job_manager.create_job(selections)
        
        assert job_id is not None
        assert job_id in job_manager.job_containers
        
        # Test job retrieval
        job_container = job_manager.get_job_container(job_id)
        assert job_container is not None
        assert job_container.job_id == job_id
        
        # Test job information
        job_info = job_manager.get_job(job_id)
        assert job_info is not None
        assert job_info["job_id"] == job_id
        
        # Test job details update
        job_manager.update_job_details(job_id, {"test_key": "test_value"})
        updated_container = job_manager.get_job_container(job_id)
        assert "test_key" in updated_container.details
        
        # Test all jobs retrieval
        all_jobs = job_manager.get_all_jobs()
        assert len(all_jobs) == 1
        assert all_jobs[0]["job_id"] == job_id
        
        print("✓ JobManager migration test passed")
        
    finally:
        if original_output_dir:
            config.OUTPUT_DIR = original_output_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_api_compatibility():
    """Test that API endpoints can work with the new architecture"""
    print("Testing API compatibility...")
    
    from vestim.backend.src.managers.job_manager import JobManager
    from vestim.backend.src.services.job_service import JobService
    
    # Create a temporary output directory
    temp_dir = tempfile.mkdtemp()
    original_output_dir = None
    
    try:
        # Temporarily override OUTPUT_DIR
        import vestim.config as config
        original_output_dir = config.OUTPUT_DIR
        config.OUTPUT_DIR = temp_dir
        
        # Create services
        job_manager = JobManager()
        job_service = JobService(job_manager)
        
        # Test job creation via service
        selections = {"model_type": "test", "data_path": "/tmp/test"}
        job_id = job_service.create_job(selections)
        
        assert job_id is not None
        
        # Test job retrieval via service
        job_info = job_service.get_job_by_id(job_id)
        assert job_info is not None
        assert job_info["job_id"] == job_id
        
        # Test job container access (new functionality)
        job_container = job_manager.get_job_container(job_id)
        assert job_container is not None
        
        # Test training task manager access
        training_manager = job_manager.get_training_task_manager(job_id)
        assert training_manager is not None
        
        print("✓ API compatibility test passed")
        
    finally:
        if original_output_dir:
            config.OUTPUT_DIR = original_output_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("JobContainer Migration Test Suite")
    print("=" * 50)
    
    try:
        test_job_container_basic()
        test_job_manager_migration()
        test_api_compatibility()
        
        print("=" * 50)
        print("✓ All tests passed! Migration successful.")
        print("=" * 50)
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"✗ Test failed: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
