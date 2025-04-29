# resonance-py: Microwave Resonator Measurement and Analysis Suite

**resonance-py** is a Python package designed to automate the measurement and analysis of microwave resonators, often used in quantum computing research and related fields. It provides tools for controlling Vector Network Analyzers (VNAs), managing measurement workflows, analyzing acquired data, and extracting key resonator parameters.

## Overview

Characterizing microwave resonators often involves multiple measurement steps, including wide-band surveys to find resonances, zoomed-in scans to determine precise parameters, and power-dependent measurements. `resonance_py` aims to streamline this process by:

* Providing instrument drivers for common Keysight VNAs (PNA N5221B, ENA E5072A/E5071B).
* Implementing an automated workflow encompassing survey, refinement, and detailed segmented scans.
* Integrating data analysis routines for artifact correction, resonator model fitting (using the inverse S21 method), and parameter extraction ($Q_i, Q_c, Q_t, f_0$).
* Offering utilities for tasks like peak finding, sweep segmentation, and logging.

## Features

* **Instrument Control:**
    * QCoDeS-style drivers for Keysight N5221B PNA and E5072A/E5071B ENA series.
    * Support for advanced VNA features like segmented sweeps.
    * Functionality to control external attenuation units via PyVISA.
* **Automated Measurement Workflow:**
    * `ResonatorMeasurement` class to orchestrate scans.
    * **Survey Scan:** Wide-band sweep to identify potential resonator frequencies.
    * **Refinement Scan:** Iteratively zooms in on identified peaks to precisely locate the resonance and estimate its FWHM.
    * **Segmented Scan:** Performs high-resolution sweeps around identified resonators, potentially at multiple power levels (attenuation settings).
* **Data Analysis:**
    * `ResonatorData` class for structured data storage.
    * `fix_artifacts`: Corrects common issues like electrical delay (phase slope) and normalization errors.
    * `single_resonator_fit`: Fits resonator data using the inverse S21 method (based on Probst/Khalil models) via `lmfit` to extract internal ($Q_i$), coupling ($Q_c$), and loaded ($Q_t$) quality factors, and resonance frequency ($f_0$).
    * SNR metrics calculation for magnitude and circle fits.
* **Data Handling & Utilities:**
    * Saving measurement data and analysis results to HDF5 files.
    * Automated plotting of scans and fits.
    * Logging utilities for tracking measurement progress and debugging.
    * Peak finding algorithms (`findPeaks`, `peak_info`).
    * Logic for generating optimized VNA sweep segments (`create_resonator_segments`).

## Installation

1.  **Prerequisites:**
    * Python version >=3.10

2.  **Clone the repository:**
    ```bash
    git clone UNDER_CONSTRUCTION
    cd resonance_py
    ```

3.  **Install Dependencies:**
    It is highly recommended to use a virtual environment.
    ```bash
    conda create -n qcodes python=3.9
    conda activate qcodes
    ```
    Create a `requirements.txt` file with the following content:
    ```txt
    numpy
    scipy
    matplotlib
    qcodes # or qcodes-core if you manage instrument drivers separately
    pyvisa
    lmfit
    h5py
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    You might also need to install `resonance_py` itself if you structure it as a package:
    ```bash
    pip install -e . # Installs in editable mode from the root directory
    ```