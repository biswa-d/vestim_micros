#!/usr/bin/env python3
"""
Comprehensive test of the VEstim multi-job dashboard architecture.
Tests all major components including job creation, status updates, and resume functionality.
"""

import sys
import os
import time
import json
import logging

# Add the vestim module to path
sys.path.insert(0, os.path.abspath('.'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_job_manager():
    """Test the JobManager functionality."""
    logger.info("Testing JobManager...")
    
    try:
        from vestim.backend.src.managers.job_manager import JobManager
        
        # Create job manager
        job_manager = JobManager()
        logger.info("✓ JobManager created")
        
        # Test job creation
        test_job_id = "test_job_001"
        success = job_manager.create_job(test_job_id, {"test_data": "sample"})
        if success:
            logger.info(f"✓ Job {test_job_id} created")
        else:
            logger.error(f"✗ Failed to create job {test_job_id}")
            return False
        
        # Test status update
        success = job_manager.update_job_status(
            test_job_id, 
            "training", 
            "training", 
            {"epoch": 1, "loss": 0.5}, 
            "Training started"
        )
        if success:
            logger.info("✓ Job status updated")
        else:
            logger.error("✗ Failed to update job status")
            return False
        
        # Test job retrieval
        job = job_manager.get_job_by_id(test_job_id)
        if job:
            logger.info(f"✓ Job retrieved: {job.get('status')}")
        else:
            logger.error("✗ Failed to retrieve job")
            return False
        
        # Test job listing
        jobs = job_manager.get_all_jobs()
        logger.info(f"✓ Retrieved {len(jobs)} total jobs")
        
        # Test resume functionality
        # First set job to a resumable status
        job_manager.update_job_status(test_job_id, "training_stopped", "training", None, "Training stopped")
        
        can_resume = job_manager.can_resume_job(test_job_id)
        if can_resume:
            logger.info("✓ Job marked as resumable")
            
            success = job_manager.resume_job(test_job_id)
            if success:
                logger.info("✓ Job resumed successfully")
            else:
                logger.error("✗ Failed to resume job")
                return False
        else:
            logger.error("✗ Job not marked as resumable")
            return False
        
        # Test resumable jobs list
        resumable_jobs = job_manager.get_resumable_jobs()
        logger.info(f"✓ Found {len(resumable_jobs)} resumable jobs")
        
        # Test deletion
        success = job_manager.delete_job(test_job_id)
        if success:
            logger.info("✓ Job deleted successfully")
        else:
            logger.error("✗ Failed to delete job")
            return False
        
        logger.info("✓ All JobManager tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ JobManager test failed: {e}")
        return False

def test_dashboard_manager():
    """Test the DashboardManager functionality."""
    logger.info("Testing DashboardManager...")
    
    try:
        from vestim.gui.src.managers.dashboard_manager import DashboardManager
        from PyQt5.QtCore import QObject
        
        # Create a minimal parent object for signals
        parent = QObject()
        
        # Create dashboard manager
        dashboard_manager = DashboardManager(parent)
        logger.info("✓ DashboardManager created")
        
        # Test server status check (this will fail if server not running, but that's expected)
        is_running = dashboard_manager.is_server_running()
        logger.info(f"✓ Server status check: {'Running' if is_running else 'Not running'}")
        
        logger.info("✓ DashboardManager basic tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ DashboardManager test failed: {e}")
        return False

def test_job_service():
    """Test the JobService functionality."""
    logger.info("Testing JobService...")
    
    try:
        from vestim.backend.src.services.job_service import JobService
        
        # Create job service
        job_service = JobService()
        logger.info("✓ JobService created")
        
        # Test job creation
        job_id, job_folder = job_service.create_new_job({"test": "data"})
        if job_id and job_folder:
            logger.info(f"✓ Job created: {job_id}")
            
            # Test status update with detailed data
            success = job_service.update_job_status(
                job_id, 
                "training", 
                "Training in progress",
                {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6}
            )
            if success:
                logger.info("✓ Job status updated with detailed data")
            else:
                logger.error("✗ Failed to update job status")
                return False
                
        else:
            logger.error("✗ Failed to create job")
            return False
        
        logger.info("✓ All JobService tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ JobService test failed: {e}")
        return False

def test_imports():
    """Test that all critical modules can be imported."""
    logger.info("Testing module imports...")
    
    modules_to_test = [
        "vestim.backend.src.managers.job_manager",
        "vestim.gui.src.managers.dashboard_manager",
        "vestim.backend.src.services.job_service",
        "vestim.backend.src.services.training_service",
        "vestim.gui.src.job_dashboard_gui_qt",
        "vestim.scripts.run_vestim"
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✓ {module_name}")
        except Exception as e:
            logger.error(f"✗ {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        logger.error(f"Failed to import {len(failed_imports)} modules")
        return False
    
    logger.info("✓ All module imports successful")
    return True

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("VEstim Multi-Job Dashboard Architecture Test")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test imports first
    logger.info("\n1. Testing Module Imports...")
    test_results.append(("Module Imports", test_imports()))
    
    # Test core components
    logger.info("\n2. Testing JobService...")
    test_results.append(("JobService", test_job_service()))
    
    logger.info("\n3. Testing JobManager...")
    test_results.append(("JobManager", test_job_manager()))
    
    logger.info("\n4. Testing DashboardManager...")
    test_results.append(("DashboardManager", test_dashboard_manager()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nOverall: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! The VEstim architecture is working correctly.")
        return True
    else:
        logger.info("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = main()
    sys.exit(0 if success else 1)
