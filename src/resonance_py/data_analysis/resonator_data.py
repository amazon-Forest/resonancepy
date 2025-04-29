# resonator_data.py
import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt

from resonance_py.data_analysis.circle_fitting import circle_fit_by_taubin
from resonance_py.data_analysis.snr_metrics import calc_mag_snr, calc_circ_snr

@dataclass
class ResonatorData:
    """Class for storing and processing resonator measurement data."""
    freq: np.ndarray
    raw_s21: np.ndarray
    s21: np.ndarray = None
    is21: np.ndarray = None
    atten: float = 0
    from_data_pathname: str = ""
    save_to_info: dict = field(default_factory=lambda: {
        "save_plots": False,  # Save plots
        "file_path": None,  # Path to save the file
        "file_type": "hdf5",  # Default file type for saving results
        "plot_name_format": "auto",  # Format for saving plots
        "plot_group": "plots",
        "show_plots": True,  # Show plots
    })
    system_attenuation: float = 70
    launch_power: float = 0
    temp: float = 0
    fit_mode: str = "probst"  # Default fit mode
    estimate: Dict[str, Any] = field(default_factory=dict)
    raw_circle: Dict[str, float] = field(default_factory=dict)
    fit: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize calculated fields after object creation."""
        if self.s21 is None:
            self.s21 = self.raw_s21.copy()
        
        # Calculate inverse S21
        self.calculate_inverse_s21()
        
        # Set default save path nametag
        if not hasattr(self, 'save_path_nametag'):
            self.save_path_nametag = "Analysis-iS21-"
        
        # Set default base filename
        if not hasattr(self, 'base_filename') and hasattr(self, 'filename'):
            self.base_filename = os.path.splitext(self.filename)[0]
            
        # Set s21 graph filename
        if hasattr(self, 'base_filename'):
            self.s21_graph_filename = f"{self.base_filename}.jpg"
            
        # Calculate raw S21 min and max for reference
        self.max_raw_s21 = 20 * np.log10(np.max(np.abs(self.raw_s21)))
        self.min_raw_s21 = 20 * np.log10(np.min(np.abs(self.raw_s21)))
    
    def calculate_inverse_s21(self):
        """Calculate inverse S21."""
        self.is21 = 1 / self.s21
    
    def calculate_estimates(self) -> Dict[str, float]:
        """
        Calculate estimated resonator parameters from S21 data.
        
        Returns:
            Dictionary of estimated resonator parameters
        """
        # Get magnitude and phase
        magnitude = np.abs(self.s21)
        phase = np.unwrap(np.angle(self.s21))
        
        # Find resonance frequency (minimum of |S21|)
        min_idx = np.argmin(magnitude)
        f0 = self.freq[min_idx]
        
        # Estimate FWHM
        half_max = (np.max(magnitude) + np.min(magnitude)) / 2
        idx_above = magnitude > half_max
        
        # Find left and right crossing points
        left_idx = np.where(idx_above[:min_idx])[0]
        right_idx = np.where(idx_above[min_idx:])[0] + min_idx
        
        # Calculate f3dB (full width at half maximum)
        if len(left_idx) > 0 and len(right_idx) > 0:
            f_left = self.freq[left_idx[-1]]
            f_right = self.freq[right_idx[0]]
            f3dB = f_right - f_left
        else:
            f3dB = (self.freq[-1] - self.freq[0]) / 10
        
        # Calculate quality factors
        qt = f0 / f3dB
        
        # Get circle parameters
        xy = np.column_stack((np.real(self.s21), np.imag(self.s21)))
        xc, yc, r = circle_fit_by_taubin(xy)
        self.ref_circle = {'xc': xc, 'yc': yc, 'r': r}
        
        # Estimate coupling phase using circle fit
        phi = np.angle(complex(xc, yc))
        
        # Estimate internal and coupling Q using diameter ratio method
        S_min = np.min(magnitude)
        S_max = np.max(magnitude)
        
        # Simple diameter method calculation
        coupling_factor = (S_max - S_min) / (S_max + S_min)
        if coupling_factor < 1:  # Undercoupled
            qc = qt / coupling_factor
            qi = qt / (1 - coupling_factor)
        else:  # Overcoupled
            qc = qt
            qi = qt * 2
        
        # Store estimates in a dictionary
        self.estimate = {
            'f0': f0,
            'f3dB': f3dB,
            'Qt': qt,
            'Qi': qi,
            'Qc': qc,
            'phi': phi,
        }
        
        return self.estimate
    
    def calculate_snr(self) -> Dict[str, float]:
        """
        Calculate SNR metrics for the data.
        
        Returns:
            Dictionary with SNR values
        """
        # Calculate SNR for direct S21
        self.s21mag_snr = calc_mag_snr(self.s21)
        self.s21circ_snr = calc_circ_snr(self.s21, True)
        
        # Calculate SNR for inverse S21
        self.is21mag_snr = calc_mag_snr(self.is21)
        self.is21circ_snr = calc_circ_snr(self.is21, False)
        
        # Return dictionary of values
        return {
            's21mag_snr': self.s21mag_snr,
            's21circ_snr': self.s21circ_snr,
            'is21mag_snr': self.is21mag_snr,
            'is21circ_snr': self.is21circ_snr
        }
    
    @classmethod
    def from_file(cls, filename: str, **kwargs) -> 'ResonatorData':
        """
        Load resonator data from a text file.
        
        Args:
            filename: Path to the file containing resonator data
            **kwargs: Additional parameters to set on the ResonatorData object
            
        Returns:
            ResonatorData object with loaded data
        """
        # Extract filename and path
        full_path = Path(filename)
        file_basename = full_path.stem
        parent_dir = str(full_path.parent)
        suffix = full_path.suffix
        
        # Load data from file - assuming file format with freq, Re(S21), Im(S21) columns
        try:
            if suffix in ['.txt', '.csv']:
                data = np.loadtxt(filename)
                freq = data[:, 0]
                re_s21 = data[:, 1]
                im_s21 = data[:, 2]
                raw_s21 = re_s21 + 1j * im_s21
                
                # Create ResonatorData object
                res_data = cls(
                    freq=freq,
                    raw_s21=raw_s21,
                    from_data_pathname=parent_dir,
                    **kwargs
                )
                
                # Set filename information
                res_data.from_filename = full_path.name
                
                
                return res_data
            elif suffix in ['.hdf5']:
                pass
            
        except Exception as e:
            raise ValueError(f"Error loading resonator data from {filename}: {e}")
    