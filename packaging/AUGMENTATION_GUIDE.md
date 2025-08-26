# VEstim Data Augmentation Guide

This guide provides an overview of the custom formula engine used for data augmentation and feature engineering in VEstim.

## Introduction

The "Create New Column" feature allows you to generate new data columns using mathematical formulas that operate on existing columns. This is a powerful tool for creating lagged features, simulating noisy sensors, smoothing data, and more.

The formula engine uses Python's `eval()` function in a controlled environment, giving you access to the pandas DataFrame, NumPy, and several custom helper functions.

## How to Use

1.  In the "Data Augmentation" window, check the "Create new columns" box.
2.  Click "Add Column Formula".
3.  In the dialog, provide a **New Column Name**.
4.  Enter your expression in the **Formula** box.
5.  Click "Create Column".

## Available Columns

You can use any of the column names present in your dataset. The list of available columns is shown in the dialog for reference. Column names are case-sensitive.

Example: If you have a column named `Battery_Temp_degC`, you must use that exact name in your formula.

## Available Functions and Operations

### 1. Basic Arithmetic

You can use standard mathematical operators:
-   `+` (Addition)
-   `-` (Subtraction)
-   `*` (Multiplication)
-   `/` (Division)
-   `**` (Exponentiation)

**Example:**
-   `new_col = Voltage * Current`

### 2. NumPy Functions

You have access to the NumPy library via the `np` prefix. This allows for a wide range of mathematical functions.

**Examples:**
-   `log_voltage = np.log(Voltage)`
-   `sin_current = np.sin(Current)`
-   `squared_power = np.power(Power, 2)`

### 3. Custom Helper Functions

Several custom functions are provided for common time-series and data augmentation tasks.

#### `noise(mean, std)`
Generates Gaussian (normal) noise.

-   `mean`: The center of the noise distribution (usually 0).
-   `std`: The standard deviation, which controls the magnitude of the noise.

**Examples:**
-   **Additive Noise**: `Temp_noisy = Battery_Temp_degC + noise(0, 0.5)` (Adds noise with a standard deviation of 0.5 degrees)
-   **Multiplicative (Percentage) Noise**: `SOC_noisy = SOC * (1 + noise(0, 0.01))` (Applies noise that is ~1% of the SOC value)

#### `shift(data_column, periods)`
Shifts a column's data up or down. This is extremely useful for creating lagged or lead features.

-   `data_column`: The name of the column to shift (e.g., `Battery_Temp_degC`).
-   `periods`: The number of steps to shift.
    -   A **negative** value (`-1`) shifts the data **up**, giving you the *next* value in the series.
    -   A **positive** value (`1`) shifts the data **down**, giving you the *previous* value in the series.

**Examples:**
-   **Lead Feature (Next Timestep)**: `Temp_next = shift(Battery_Temp_degC, -1)`
-   **Lagged Feature (Previous Timestep)**: `Power_previous = shift(Power, 1)`

#### `moving_average(data_column, window)`
Calculates the simple moving average of a column.

-   `data_column`: The name of the column to process.
-   `window`: The number of samples to include in the moving average window.

**Example:**
-   `Power_smoothed = moving_average(Power, 100)` (Calculates the 100-sample moving average of the `Power` column)

#### `rolling_max(data_column, window)`
Calculates the rolling maximum of a column.

-   `data_column`: The name of the column to process.
-   `window`: The number of samples to include in the rolling window.

**Example:**
-   `Peak_Current_last_50 = rolling_max(Current, 50)`

### 4. Advanced Examples (Combining Functions)

You can combine these functions to create more complex and realistic features.

**Example: Simulating a Drifting Sensor**
This formula simulates an SOC sensor that has a slow, drifting error, which is more realistic than pure random noise.

-   **New Column Name**: `SOC_drifted`
-   **Formula**: `SOC * (1 + moving_average(noise(0, 0.01), 500))`

**How it works:**
1.  `noise(0, 0.01)` creates a random signal.
2.  `moving_average(..., 500)` smooths this random signal over a 500-sample window, turning it into a slow, wandering "drift".
3.  `SOC * (1 + ...)` applies this drifting signal as a multiplicative error to the original `SOC`.