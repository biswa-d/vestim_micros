import os
from vestim.config_manager import get_output_directory

# Define the root of your project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the output directory - use installer config if available, otherwise default
OUTPUT_DIR = get_output_directory()

