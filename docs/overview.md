# `resonance-py`: Superconducting Resonator Measurement and Analysis

## Overview

`resonance-py` is a Python package designed for controlling laboratory equipment, performing measurements on superconducting resonators, and analyzing the resulting data. The package provides tools for:

1. **Instrument Control**: Drivers for communicating with Vector Network Analyzers (VNAs) like the Keysight E5072A and N5221B
2. **Measurement Workflows**: Structured processes for survey scans, refinement scans, and segmented measurements
3. **Data Analysis**: Sophisticated fitting routines for resonator data, including circle fitting and parameter extraction
4. **Data Visualization**: Plotting tools for presenting measurement results and fits

## Key Features

- **Complete Measurement Pipeline**: From initial survey scans to refined measurements with segmented frequency ranges
- **Flexible Resonator Fitting**: Implementation of various fitting models (Probst, inverse S21) for accurately characterizing resonators
- **Instrument Abstraction**: High-level interfaces to laboratory equipment
- **SNR Calculation**: Methods for determining signal-to-noise ratio in resonator measurements
- **Data Management**: Utilities for saving and loading measurement data and plots

## Typical Workflow

1. Initialize connection to VNA instrument
2. Perform a survey scan to identify resonator frequencies
3. Run refinement scans on each identified resonator
4. Perform segmented measurements at different attenuation levels
5. Analyze data by fitting to resonator models
6. Extract key parameters (Q factors, resonant frequencies)
7. Save and visualize results

## Target Audience

This package is designed for researchers and engineers working with superconducting resonators, particularly for quantum device characterization, circuit QED experiments, and materials characterization.

## Getting Started

See the [User Guide](user_guide.md) for installation instructions and tutorials on how to use the package.