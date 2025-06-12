import sys
import os

# Add the project root to the Python path, assuming 'vestim_micros' is the root
# and this script is in 'vestim_micros/test_import.py'
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Sys.path: {sys.path}")

try:
    from vestim.backend.src.services.job_service import JobService
    print("Successfully imported JobService from vestim.backend.src.services.job_service")
    js = JobService()
    print("Successfully instantiated JobService")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    import vestim.config
    print("Successfully imported vestim.config")
except ImportError as e:
    print(f"Failed to import vestim.config: {e}")

try:
    import vestim.logger_config
    print("Successfully imported vestim.logger_config")
except ImportError as e:
    print(f"Failed to import vestim.logger_config: {e}")

# Attempt to mimic main.py's import structure more closely
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
# This path adjustment above is for files deep within backend/src, not for a root script.
# The initial sys.path.insert(0, project_root) should be correct if 'vestim_micros' is the intended root.

