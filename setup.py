#!/usr/bin/env python3
"""
VESTim Regression GPU Package Setup
===================================
This setup script creates an installable package for VESTim.
"""

import os
import re
from setuptools import setup, find_packages

# Get the current directory
here = os.path.abspath(os.path.dirname(__file__))

# Get the version from the code
with open(os.path.join(here, 'vestim', '__init__.py'), 'r') as f:
    init_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file, re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = '0.1.0'  # Default if not found

# Get the long description from the README file
try:
    with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = "VESTim - A GPU-Accelerated Regression Tool for Battery State Estimation"

# Define package requirements
requirements = [
    'torch>=1.8.0',
    'pandas>=1.2.0',
    'numpy>=1.20.0',
    'PyQt5>=5.15.0',
    'matplotlib>=3.3.0',
    'scikit-learn>=0.24.0',
    'configparser>=5.0.0',
]

# Define dev requirements
dev_requirements = [
    'pytest>=6.0.0',
    'black>=21.5b0',
    'flake8>=3.9.0',
]

setup(
    name="vestim-reg-gpu",
    version=version,
    description="VESTim - GPU-Accelerated Regression Tool for Battery State Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vestim-gpu",
    author="Biswanath Dehury",
    author_email="your.email@example.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="battery, regression, machine learning, pytorch, gpu",
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "vestim=launcher:main",
            "vestim-gui=vestim.gui.src.main_gui:main",
            "vestim-data=vestim.gui.src.data_import_gui_qt:main",
            "vestim-train=vestim.gui.src.training_setup_gui_qt:main",
            "vestim-test=vestim.gui.src.testing_gui_qt:main",
            "vestim-config=vestim.config.create_config:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vestim": ["config/*.json", "gui/resources/*"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/vestim-gpu/issues",
        "Source": "https://github.com/yourusername/vestim-gpu",
    },
)
