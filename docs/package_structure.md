# Package Structure

This document provides a detailed overview of the `resonance-py` package structure and explains the purpose of each module.

### `data_analysis` Sub-package

Contains modules for processing, analyzing, and visualizing resonator data.

- `artifacts.py`: Functions for identifying and correcting artifacts in measurement data
- `circle_fitting.py`: Algorithms for fitting resonator data to a circle in the complex plane
- `modeling.py`: Models used for resonator behavior simulation and fitting
  - Contains `single_resonator_fit`, `res_is21_fit`, `probst_fit` and visualization functions
- `resonator_data.py`: Classes for managing resonator measurement data
- `res_iS21_fit_function.py`: Inverse S21 fitting implementation with `lmfit`
- `snr_metrics.py`: Functions for calculating Signal-to-Noise Ratio metrics

### `drivers` Sub-package

Device drivers for controlling laboratory equipment.

- `keysightE5072A.py`: Driver for Keysight E5072A VNA
- `keysightN5221B.py`: Driver for Keysight N5221B VNA
- `SetAttenuation.py`: Utility for controlling system attenuation
- `vna_manuals/`: Reference materials for VNA instruments

### `measurements` Sub-package

Defines measurement sequences and workflows.

- `resonatorMeasurement.py`: Main class for performing resonator measurements
  - Implements survey, refinement, and segmented measurement workflows

### `simulation` Sub-package

- (Currently empty - planned for future implementation)

### `utils` Sub-package

Utility functions used throughout the package.

- `logging_utils.py`: Utilities for setting up logging
- `peak_analysis.py`: Functions for finding and analyzing spectral peaks
- `segmentation.py`: Tools for creating segmented frequency sweeps
- `statistics.py`: Statistical utilities for data processing

## File Organization

```
resonance-py/
├── data/                  # Default directory for measurement data
├── docs/                  # Documentation
├── notebooks/            # Jupyter notebooks with examples
├── src/
│   └── resonance_py/     # Main package
│       ├── __init__.py
│       ├── data_analysis/
│       ├── drivers/
│       ├── measurements/
│       ├── simulation/
│       └── utils/
└── tests/                # Test modules
```

## Module Dependencies

The core package dependencies include:
- NumPy: Numerical operations
- Matplotlib: Visualization
- QCoDeS: Instrument control framework
- h5py: Data storage
- lmfit: Curve fitting
- SciPy: Signal processing and optimization
