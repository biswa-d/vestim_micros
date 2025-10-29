# Vestim User Guide

Welcome to Vestim! This guide will walk you through the process of using the Vestim application to train and test your own models.

## Getting Started

1.  **Launch the Application**: Double-click the Vestim shortcut on your desktop or in the Start Menu.
2.  **Data Import**: Use the Data Import screen to select your training, validation, and testing data.
3.  **Data Augmentation**: Prepare your data for training by applying augmentation techniques.
4.  **Hyperparameters**: Configure your model and training settings.
5.  **Training**: Train your model and monitor its progress.
6.  **Testing**: Evaluate your model's performance on the test data.

## Data Augmentation

The Data Augmentation screen allows you to apply various transformations to your data before training.

### Data Filtering

You can apply several types of filters to your data to smooth it or remove noise. The available filters are:

*   **Butterworth**: A type of signal processing filter designed to have a frequency response that is as flat as possible in the passband.
*   **Savitzky-Golay**: A filter that smooths data by fitting successive sub-sets of adjacent data points with a low-degree polynomial by the method of linear least squares.
*   **Exponential Moving Average (EMA)**: A type of moving average that places a greater weight and significance on the most recent data points.

### Column Creation (Formulas)

You can create new columns in your dataset using formulas. This is useful for feature engineering. The formula engine supports standard arithmetic operators, NumPy functions, and several built-in functions.

**Examples:**

*   `column1 * 2 + column2`
*   `np.sin(column1) + np.log(column2)`
*   Absolute noise: `column1 + noise(0.0, 0.02)`
*   Relative noise: `column1 * (1 + noise(0.0, 0.02))`
*   Moving average: `moving_average(column1, 10)`
*   Rolling max: `rolling_max(column1, 10)`

## Troubleshooting (Windows)

If PyBattML fails to start with an error mentioning `c10.dll` or `WinError 1114` when importing PyTorch, follow these steps:

1) Install or Repair Microsoft Visual C++ 2015–2022 Redistributable (x64)
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Run the installer and complete setup. Reboot if prompted.

2) Ensure CPU‑only PyTorch is installed
- If you’re on a CPU‑only machine, PyTorch should be installed from the CPU wheel index.
- If you previously installed a CUDA wheel, uninstall `torch`, `torchvision`, and `torchaudio` and reinstall them using the CPU index URL.

3) Relaunch PyBattML

Notes:
- On some systems, corporate policies or being offline may block the installer from installing VC++ automatically. Installing VC++ manually then rerunning the installer resolves the issue.
- Very old CPUs without AVX support cannot load modern PyTorch wheels; contact support if you suspect this.
