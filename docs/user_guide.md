# User Guide

This guide provides instructions for using the `resonance-py` package for superconducting resonator measurements and analysis.

## Installation

To install the package:

```bash
git clone https://github.com/username/resonance-py.git
cd resonance-py
pip install -e .
```

## Basic Usage

### Setting Up Instrument Connection

```python
from resonance_py.drivers.keysightN5221B import KeysightN5221B

# Connect to the VNA instrument
pna = KeysightN5221B("pna", "TCPIP0::192.168.1.1::inst0::INSTR")
```

### Creating a ResonatorMeasurement Instance

```python
from resonance_py.measurements.resonatorMeasurement import ResonatorMeasurement

# Define measurement settings
settings = {
    'sampleName': 'Sample1',
    'centerFrequency': 5e9,  # 5 GHz
    'frequencySpan': 1e9,    # 1 GHz
    'numOfResonators': 3,
    'selectedResonators': [0, 1, 2], 
    'attenuationValues': [0, 10, 20],
    'systemAttenuation': 30,
    'survey': {
        'points': 1001,
        'if_bandwidth': 1000,
        'measurement': 'S21',
        'save_data': True,
    },
    'refine': {
        'points': 501,
        'if_bandwidth': 100,
        'save_data': True,
    },
    'segment': {
        'averaging': 3,
        'save_data': True,
    }
}

# Initialize the measurement controller
measurement = ResonatorMeasurement(
    pna=pna,
    save_path='./data/my_experiment',
    save_base_name='resonator_test',
    settings=settings
)
```

### Running a Complete Measurement

```python
# Perform the full measurement sequence
measurement.full_scan(save_data=True)
```

### Running Individual Measurement Steps

For more control, you can run each step separately:

```python
# 1. Run the survey scan to find resonator frequencies
measurement.survey_scan(plot=True)

# 2. Refine measurements for a specific resonator
refined_data, all_refined = measurement.refinement_scan(resonator_index=0, plot=True)

# 3. Run segmented measurements at different attenuation values
segment_data = measurement.segmented_scan(refined_data, plot=True, analyze=True)
```

## Data Analysis

### Using the ResonatorData Class

```python
from resonance_py.data_analysis.resonator_data import ResonatorData
from resonance_py.data_analysis.modeling import single_resonator_fit, resonator_fit_plot

# Create a ResonatorData object from frequency and S21 data
resonator_data = ResonatorData(
    freq=frequencies,
    raw_s21=s21_complex_data,
    atten=30,  # Attenuation value in dB
)

# Set save information for plots
resonator_data.save_to_info = {
    "save_plots": True,
    "file_type": "png",
    "plot_name_format": 'manual',
    "plot_group": "plots",
    "show_plots": True,
    'file_path': './data/plots',
    'plot_name': 'resonator_fit',
}

# Fit the resonator data
resonator_data = single_resonator_fit(resonator_data, opts={"plot": True})

# Display fit results
print(f"Internal Q: {resonator_data.fit['Qi']:.2e}")
print(f"Coupling Q: {resonator_data.fit['Qc']:.2e}")
print(f"Total Q: {resonator_data.fit['Qt']:.2e}")
print(f"Resonant frequency: {resonator_data.fit['fo']/1e9:.6f} GHz")
```

## Working with Instrument Drivers Directly

### Performing Linear Sweeps

```python
# Perform a simple linear frequency sweep
complex_data, frequencies = pna.linear_sweep(
    center_frequency=5e9,    # 5 GHz 
    frequency_span=100e6,    # 100 MHz
    points=1001, 
    if_bandwidth=1000,       # 1 kHz
    averaging=1,
    measurement='S21'
)
```

### Performing Segmented Sweeps

```python
# Define segments for focused measurement
segments = [
    {
        'start': 5.0e9,      # 5.0 GHz
        'stop': 5.01e9,      # 5.01 GHz
        'points': 101,
        'ifbw': 100          # 100 Hz for fine detail
    },
    {
        'start': 5.02e9,     # 5.02 GHz
        'stop': 5.03e9,      # 5.03 GHz
        'points': 101,
        'ifbw': 100          # 100 Hz for fine detail
    }
]

# Perform a segmented sweep
complex_data, frequencies = pna.segmented_sweep(
    segments_data=segments,
    averaging=3,
    measurement='S21'
)
```

## Logging

The package includes a comprehensive logging system:

```python
from resonance_py.utils.logging_utils import setup_logger, log_exception

# Setup a logger
logger = setup_logger(
    name='my_experiment',
    log_file='./data/experiment.log',
    level=logging.INFO
)

# Log information
logger.info("Starting measurement")

# Log errors with traceback
try:
    # Measurement code
    pass
except Exception as e:
    log_exception(logger, exc_info=True)
```

## Examples

For more detailed examples, refer to the Jupyter notebooks in the `notebooks` directory:
- `data_analysis.ipynb`: Examples of data analysis workflows
- `demo.ipynb`: Walks through the entire workflow
- `qcodesTesting.ipynb`: Working with QCoDeS for instrument control