import sys
import os
import traceback

# This script assumes it is located in the 'vestim_micros' project root directory.
# Add the project root to sys.path to ensure 'from vestim...' imports work correctly.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Current working directory: {os.getcwd()}")
print(f"Sys.path: {sys.path}")
print("-" * 50)

try:
    print("Attempting: from vestim.backend.src.services.job_service import JobService")
    from vestim.backend.src.services.job_service import JobService
    print("Successfully imported JobService.")
    # If import is successful, try to instantiate
    # print("Attempting: js = JobService()")
    # js = JobService()
    # print("Successfully instantiated JobService.")
except ImportError as e:
    print(f"Caught ImportError: {e}")
    print("--- Traceback for ImportError ---")
    traceback.print_exc()
    print("-" * 50)
except Exception as e:
    print(f"Caught other Exception during import process: {type(e).__name__}: {e}")
    print("--- Traceback for other Exception ---")
    traceback.print_exc()
    print("-" * 50)

# For further diagnosis, let's try importing the module itself
try:
    print("\nAttempting: import vestim.backend.src.services.job_service as js_module")
    import vestim.backend.src.services.job_service as js_module
    print("Successfully imported vestim.backend.src.services.job_service as js_module.")
    print("Attributes in js_module:", dir(js_module))
except ImportError as e:
    print(f"Caught ImportError when importing module: {e}")
    print("--- Traceback for ImportError (module import) ---")
    traceback.print_exc()
    print("-" * 50)
except Exception as e:
    print(f"Caught other Exception during module import: {type(e).__name__}: {e}")
    print("--- Traceback for other Exception (module import) ---")
    traceback.print_exc()
    print("-" * 50)
