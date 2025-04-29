import numpy as np
from typing import Union, Optional
from .circle_fitting import circle_fit_by_taubin

def calc_mag_snr(s21: np.ndarray) -> float:
    """
    Calculate magnitude signal-to-noise ratio of S21 data.
    
    Args:
        s21: Complex S21 data array
        
    Returns:
        Magnitude SNR value
    """
    magnitude = np.abs(s21)
    
    # Total signal range (max - min) across whole dataset
    signal_range = np.max(magnitude) - np.min(magnitude)
    
    # Range of first 10 points as noise estimate
    noise_estimate = np.max(magnitude[:10]) - np.min(magnitude[:10])
    
    # Avoid division by zero
    if noise_estimate > 0:
        return signal_range / noise_estimate
    else:
        return float('inf')  # Return infinity if noise is zero

def calc_circ_snr(s21: np.ndarray, s21_mode: int = 0) -> tuple[float, float]:
    """
    Calculate circular signal-to-noise ratio by fitting S21 data to a circle.
    
    Args:
        s21: Complex S21 data array
        s21_mode: 0 for inverse S21 (max at resonance), 1 for regular S21 (min at resonance)
        
    Returns:
        Tuple of (overall circular SNR, 3dB bandwidth circular SNR)
    """
    # Extract real and imaginary parts
    x = np.real(s21)
    y = np.imag(s21)
    
    # Combine into XY points for circle fitting
    xy_points = np.column_stack((x, y))
    
    # Fit circle to the points using Taubin's method
    xc, yc, r = circle_fit_by_taubin(xy_points)
    
    # Calculate distances from each point to the fitted circle
    d = np.sqrt((x - xc)**2 + (y - yc)**2)
    
    # Calculate standard deviation of distances (noise)
    std = np.sqrt(np.sum((d - r)**2) / len(x))
    
    # Full circle SNR
    circ_snr = r / std if std > 0 else float('inf')
    
    # Find the resonance point (minimum or maximum depending on mode)
    if s21_mode == 0:  # Inverse S21 mode
        fo_idx = np.argmax(np.abs(s21))
    else:  # Regular S21 mode
        fo_idx = np.argmin(np.abs(s21))
    
    # Calculate 3dB point
    three_db_point = 0.5 * (np.min(np.abs(s21)) + np.max(np.abs(s21)))
    
    # Find indices of 3dB points
    fn3db_idx = np.argmin(np.abs(np.abs(s21[:fo_idx]) - three_db_point))
    fp3db_idx = np.argmin(np.abs(np.abs(s21[fo_idx:]) - three_db_point))
    fp3db_idx = fp3db_idx + fo_idx - 1
    
    # Calculate SNR using only points within 3dB bandwidth
    x_mod = x[fn3db_idx:fp3db_idx+1]
    y_mod = y[fn3db_idx:fp3db_idx+1]
    d_mod = np.sqrt((x_mod - xc)**2 + (y_mod - yc)**2)
    sd_mod = np.sqrt(np.sum((d_mod - r)**2) / len(x_mod))
    
    # 3dB bandwidth SNR
    circ_snr_f3db = r / sd_mod if sd_mod > 0 else float('inf')
    
    return circ_snr, circ_snr_f3db