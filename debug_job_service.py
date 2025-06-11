#!/usr/bin/env python3
"""Debug script to test JobService class definition"""

print("Starting debug script...")

try:
    print("Testing imports...")
    import os
    import json
    import itertools
    from datetime import datetime
    print("Basic imports successful")
    
    from vestim.config import OUTPUT_DIR
    print(f"OUTPUT_DIR imported: {OUTPUT_DIR}")
    
    from vestim.logger_config import configure_job_specific_logging
    print("configure_job_specific_logging imported")
    
    import logging
    import glob
    import shutil
    print("All imports successful")
    
    print("Defining JobService class...")
    
    class JobService:
        _instance = None

        def __new__(cls, *args, **kwargs):
            if not cls._instance:
                cls._instance = super(JobService, cls).__new__(cls)
            return cls._instance

        def __init__(self):
            if not hasattr(self, 'initialized'):
                self.initialized = True
                print("JobService initialized")
    
    print("JobService class defined successfully")
    
    # Test instantiation
    job_service = JobService()
    print(f"JobService instance created: {job_service}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
