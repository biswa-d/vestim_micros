from setuptools import setup, find_packages

setup(
    name="vestim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "requests>=2.26.0",
        "PyQt5>=5.15.0",
        "torch>=1.9.0",
        "pydantic>=1.8.0",
    ],    entry_points={
        'console_scripts': [
            'vestim=vestim.scripts.entrypoint:main',
            'vestim-gui=vestim.scripts.run_gui:main',
            'vestim-server=vestim.scripts.run_server:main',
        ],
    },
    python_requires=">=3.7",
    author="Biswanath Dehury",
    author_email="your.email@example.com",
    description="A tool for ML model training with client-server architecture",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
