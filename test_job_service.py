#!/usr/bin/env python3
"""Test script for debugging the JobService/JobManager issue"""

try:
    print("Testing JobService import...")
    from vestim.backend.src.services.job_service import JobService
    print("✓ JobService imported successfully")
    
    print("Testing JobService instantiation...")
    js = JobService()
    print("✓ JobService instantiated successfully")
    print(f"Type: {type(js)}")
    
    print("Testing get_all_jobs method...")
    print(f"Has get_all_jobs: {hasattr(js, 'get_all_jobs')}")
    jobs = js.get_all_jobs()
    print(f"✓ get_all_jobs() works, returned: {type(jobs)}")
    
    print("Testing main.py dependency injection...")
    from vestim.backend.src.main import get_job_service
    js2 = get_job_service()
    print(f"✓ main.py get_job_service() works, returned: {type(js2)}")
    print(f"Has get_all_jobs: {hasattr(js2, 'get_all_jobs')}")
    
    jobs2 = js2.get_all_jobs()
    print(f"✓ get_all_jobs() from main.py works, returned: {type(jobs2)}")
    
    print("All tests passed! The JobService is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
