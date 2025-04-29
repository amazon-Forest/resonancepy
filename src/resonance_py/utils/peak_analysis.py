from .statistics import stats_of_data

import numpy as np
from scipy.signal import find_peaks, detrend, peak_widths
from typing import Dict, List, Union, Tuple, Any

def findPeaks(
        freq: np.ndarray,
        data: np.ndarray,
        peak_method: str = 's21Mag_gradient',
        expectedPeaks: int = 5,
        stdMult: float = 2,
        distance: int = 2,
        is_dip: bool = True
    ) -> np.ndarray:
    """
    Find peaks in the provided data.
    
    Parameters:
    -----------
    freq : np.ndarray
        Frequency array
    data : np.ndarray
        Data values where peaks will be found
    peak_method : str
        Method to find peaks. Currently supported: 's21Mag_gradient'
    expectedPeaks : int
        Number of peaks expected to find
    stdMult : float
        Multiplier for standard deviation threshold
    distance : int
        Minimum distance between peaks
    
    Returns:
    --------
    np.ndarray
        Indices of detected peaks
    """
    try:
        if peak_method == 's21Mag_gradient':
            if expectedPeaks > 1:
                dData = np.gradient(data)
                std = stats_of_data(abs(dData))['std'] * stdMult
                peaks, _ = find_peaks(dData, height=std, distance=distance)

                while len(peaks) > expectedPeaks:
                    std *= stdMult
                    peaks, _ = find_peaks(dData, height=std, distance=distance)
                
                return peaks - 1 if len(peaks) > 0 else np.array([])
            else:
                if not is_dip:
                    return np.array([np.argmax(data)])
                else:
                    return np.array([np.argmin(data)])
                
        else:
            valid_methods = ['s21Mag_gradient']
            raise ValueError(f"Invalid peak method '{peak_method}'. Valid options are: {valid_methods}")
        
    except Exception as e:
        print(f"Error finding peaks: {e}")
        return np.array([])


def _fix_zero_width_peak(
    data: np.ndarray,
    peak_idx: int,
    window_size: int = 10,
    is_dip: bool = True
) -> int:
    """
    Fix a zero-width peak by finding a better local extremum in a window around the peak.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array containing the peak
    peak_idx : int
        Index of the problematic peak
    window_size : int
        Size of window around the peak to search for better extremum
    is_dip : bool
        Whether we're looking for a minimum (dip) or maximum (peak)
    
    Returns:
    --------
    int
        New peak index within the window
    """
    # Create a window around the peak, respecting array boundaries
    start_idx = max(0, peak_idx - window_size // 2)
    end_idx = min(len(data), peak_idx + window_size // 2 + 1)
    
    window = data[start_idx:end_idx]
    
    # Find local extremum within the window
    if is_dip:
        # For dips (resonators), find minimum
        local_extremum_idx = np.argmin(window)
    else:
        # For peaks, find maximum
        local_extremum_idx = np.argmax(window)
    
    # Convert back to original data index
    new_peak_idx = start_idx + local_extremum_idx
    
    return new_peak_idx


def peak_info(
    data: np.ndarray,
    frequencies: np.ndarray,
    expectedPeaks: int = 5,
    peak_method: str = 's21Mag_gradient',
    peakIndex: np.ndarray = None,
    rel_height: float = 0.5,
    min_width_points: float = 1.0,
    fix_zero_peaks: bool = True,
    window_size: int = 10,
    is_dip: bool = True
) -> Dict[str, np.ndarray]:
    """
    Calculate Full Width at Half Maximum (FWHM) for peaks in the data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data values containing peaks
    frequencies : np.ndarray
        Frequency array corresponding to data
    expectedPeaks : int
        Number of peaks expected to find
    peak_method : str
        Method to find peaks if peakIndex not provided
    peakIndex : np.ndarray, optional
        Pre-calculated peak indices
    rel_height : float
        Relative height for width calculation (0.5 = half maximum)
    min_width_points : float
        Minimum width in data points to use when zero width is detected
    fix_zero_peaks : bool
        Whether to attempt to fix zero-width peaks by finding better local extrema
    window_size : int
        Size of window to use when fixing zero-width peaks
    is_dip : bool
        Whether peaks are actually dips (resonators, True) or actual peaks (False)
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing peak information including indices, widths,
        width heights, and Q factors
    """
    # Initialize peaks_info
    peaks_info = {}
    
    # Find peaks if not provided
    if peakIndex is None:
        peakIndex = findPeaks(
            freq=frequencies,
            data=data,
            peak_method=peak_method,
            expectedPeaks=expectedPeaks,
            is_dip=is_dip

        )
    
    # if isdip is True, then we need to invert the data to calculate widths
    if is_dip:
        peakData = -data
    else:
        peakData = data

    # If we have peaks, calculate widths
    if len(peakIndex) > 0:
        # Fix zero-width peaks by finding better peak indexes.  Used refined peaks for width calculation
        if fix_zero_peaks:
            # First attempt: try to find better peak positions before width calculation
            refined_peaks = np.array([
                _fix_zero_width_peak(data, idx, window_size, is_dip) 
                for idx in peakIndex
            ])
            
            # Use refined peaks for width calculation
            widths, width_height, left_ips, right_ips = peak_widths(
                peakData, refined_peaks, rel_height=rel_height
            )
            
            # Keep track of which peaks were refined
            original_peaks = peakIndex.copy()
            peakIndex = refined_peaks
        else:
            # Standard width calculation
            widths, width_height, left_ips, right_ips = peak_widths(
                peakData, peakIndex, rel_height=rel_height
            )

        # Fix any remaining zero widths with minimum width and adjust interpolated points
        zero_widths_count = np.sum(widths == 0)
        if zero_widths_count > 0:
            print(f"Warning: {zero_widths_count} peaks have width of 0 after refinement, applying minimum width of {min_width_points} points")
            
            for i, width in enumerate(widths):
                if (width == 0):
                    # For zero width, set to minimum and adjust interpolated points
                    widths[i] = min_width_points
                    idx = peakIndex[i]
                    # Center the width around the peak
                    left_ips[i] = idx - min_width_points/2
                    right_ips[i] = idx + min_width_points/2

        peaks_info = {
            'peakIndex': peakIndex,
            'widths': widths,
            'width_heights': width_height,
            'left_ips': left_ips,
            'right_ips': right_ips
        }
        
        # If peaks were refined, add the original peaks to the info
        if fix_zero_peaks:
            peaks_info['original_peakIndex'] = original_peaks

        # Calculate FWHM, resonant frequency, and quality factor with better interpolation
        fwhm_freq = []
        fo = []
        q = []
        left_freqs = []  # New array to store left edge frequencies
        right_freqs = []  # New array to store right edge frequencies
        
        for idx, left_ip, right_ip in zip(
            peaks_info['peakIndex'],
            peaks_info['left_ips'],
            peaks_info['right_ips']
        ):
            # get a better interpolation for left frequency
            if left_ip < 0:
                left_freq = frequencies[0]
            else:
                left_idx_floor = int(np.floor(left_ip))
                left_idx_ceil = int(np.ceil(left_ip))
                
                # Ensure indices are within bounds
                left_idx_floor = max(0, min(left_idx_floor, len(frequencies) - 1))
                left_idx_ceil = max(0, min(left_idx_ceil, len(frequencies) - 1))
                
                if left_idx_floor == left_idx_ceil:
                    left_freq = frequencies[left_idx_floor]
                else:
                    # Linear interpolation
                    frac = left_ip - left_idx_floor
                    left_freq = frequencies[left_idx_floor] * (1 - frac) + frequencies[left_idx_ceil] * frac
            
            # Better interpolation for right frequency
            if right_ip >= len(frequencies):
                right_freq = frequencies[-1]
            else:
                right_idx_floor = int(np.floor(right_ip))
                right_idx_ceil = int(np.ceil(right_ip))
                
                # Ensure indices are within bounds
                right_idx_floor = max(0, min(right_idx_floor, len(frequencies) - 1))
                right_idx_ceil = max(0, min(right_idx_ceil, len(frequencies) - 1))
                
                if right_idx_floor == right_idx_ceil:
                    right_freq = frequencies[right_idx_floor]
                else:
                    # Linear interpolation
                    frac = right_ip - right_idx_floor
                    right_freq = frequencies[right_idx_floor] * (1 - frac) + frequencies[right_idx_ceil] * frac
            
            # Store the interpolated frequencies
            left_freqs.append(left_freq)
            right_freqs.append(right_freq)
            
            # Calculate width in frequency units (ensure positive width)
            fwhm_freq_cur = max(abs(right_freq - left_freq), 
                              (frequencies[1] - frequencies[0]) * min_width_points/2)  # Minimum frequency width
            fo_curr = frequencies[idx]
            
            # Calculate Q factor (now division by zero is prevented)
            q_curr = fo_curr / fwhm_freq_cur
            
            fwhm_freq.append(fwhm_freq_cur)
            fo.append(fo_curr)
            q.append(q_curr)
        
        peaks_info['fwhm_freq'] = np.array(fwhm_freq)
        peaks_info['fo'] = np.array(fo)
        peaks_info['q'] = np.array(q)
        peaks_info['left_freq'] = np.array(left_freqs)  # Add left edge frequencies to output
        peaks_info['right_freq'] = np.array(right_freqs)  # Add right edge frequencies to output
    else:
        # Return empty arrays if no peaks found
        peaks_info = {
            'peakIndex': np.array([]),
            'widths': np.array([]),
            'width_heights': np.array([]),
            'left_ips': np.array([]),
            'right_ips': np.array([]),
            'fwhm_freq': np.array([]),
            'fo': np.array([]),
            'q': np.array([]),
            'left_freq': np.array([]), 
            'right_freq': np.array([]) 
        }

    return peaks_info