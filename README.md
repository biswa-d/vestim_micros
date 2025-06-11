# VEstim - ML Model Training Tool

VEstim is a machine learning model training tool designed specifically for battery and supercapacitor voltage estimation. It provides a complete workflow from data import to model training and evaluation, with a user-friendly interface.

## New Client-Server Architecture

VEstim now uses a client-server architecture that provides several benefits:

1. **Resilience**: Training continues even if the GUI crashes
2. **Persistence**: Jobs and tasks are tracked properly
3. **Monitoring**: Multiple jobs can be tracked through the dashboard
4. **Separation of Concerns**: Frontend and backend are properly decoupled

See [CLIENT_SERVER_GUIDE.md](CLIENT_SERVER_GUIDE.md) for detailed instructions on using the new architecture.

## Features

- **Data Import**: Import and preprocess data from various sources (Arbin, STLA, Digatron, etc.)
- **Data Augmentation**: Enhance your dataset with resampling and feature engineering
- **Hyperparameter Selection**: Tune model hyperparameters for optimal performance
- **Model Training**: Train LSTM models with automatic parameter tuning
- **Model Evaluation**: Evaluate model performance with comprehensive metrics
- **Job Dashboard**: Monitor and manage multiple training jobs

## Installation

### Quick Installation

Using the provided setup script:

#### Windows
```cmd
vestim_setup.cmd install
```

#### Linux/Mac
```bash
chmod +x vestim_setup.sh
./vestim_setup.sh install
```

### Manual Installation

```bash
pip install -e .
```

## Usage

### Starting VEstim

Start the server and GUI:

```bash
vestim all
```

Or start components separately:

```bash
# Start the server
vestim server

# Start the GUI
vestim gui
```

### Workflow

1. **Data Import**: Select your data files and preprocess them
2. **Data Augmentation**: Enhance your dataset as needed
3. **Hyperparameter Selection**: Configure your model parameters
4. **Training**: Train your model with the selected parameters
5. **Evaluation**: Evaluate model performance on test data

## Development

### Project Structure

- `vestim/backend/`: Server-side code
  - `src/main.py`: FastAPI backend entry point
  - `src/services/`: Core services (job management, training, etc.)
- `vestim/gui/`: Client-side GUI code
  - `src/`: PyQt5 GUI components
- `vestim/scripts/`: Command-line tools and entry points

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Requirements

- Python 3.9+
- PyTorch
- FastAPI
- PyQt5
- Other dependencies listed in requirements.txt

## License

Copyright © 2025 Biswanath Dehury

## Contact

For questions or support, please contact the development team.