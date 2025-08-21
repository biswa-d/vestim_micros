import os
import sys

# This hook is executed at runtime, before the main script.
# It's used to configure the environment for the bundled application.

if sys.platform == 'win32':
    # The _MEIPASS environment variable is set by PyInstaller.
    # It contains the path to the temporary folder where the app is unpacked.
    if hasattr(sys, '_MEIPASS'):
        # Add the bundled torch/lib directory to the DLL search path.
        # This ensures that dependencies like fbgemm.dll can be found.
        torch_lib_path = os.path.join(sys._MEIPASS, 'torch', 'lib')
        if os.path.isdir(torch_lib_path):
            os.add_dll_directory(torch_lib_path)