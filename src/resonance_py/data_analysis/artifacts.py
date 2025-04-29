# artifacts.py
import numpy as np
from scipy import signal, optimize
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import lmfit  # Add import for lmfit

from resonance_py.data_analysis.resonator_data import ResonatorData
from resonance_py.data_analysis.circle_fitting import circle_fit_by_taubin
from resonance_py.data_analysis.snr_metrics import calc_mag_snr, calc_circ_snr
from resonance_py.utils.peak_analysis import peak_info

def fix_artifacts(resonator_data: ResonatorData, options: Optional[Dict[str, Any]] = None) -> ResonatorData:
    """
    Main function to correct artifacts in S21 data.
    
    Args:
        resonator_data: ResonatorData object containing S21 measurements
        options: Dictionary of correction options
        
    Returns:
        ResonatorData object with corrected S21 data
    """
    # Set default options
    opts = {
        "initial_corrects": True,
        "shrink_freq_range": True,
        "phase_slope": True,
        "equate_phase_ends": False, 
        "remove_phase_spiral": False,
        "remove_fit_phase_errors": True,
        "remove_fit_amplitude_errors": True,
        "plot_save_enable": False,
        "debug_artifacts": False,
        "fit_mode": "lmfit",  # New option to control fitting method
    }
    
    # Update with user options
    if options:
        opts.update(options)
        
    # Store original data
    resonator_data.raw_freq = resonator_data.freq.copy()
    resonator_data.raw_s21 = resonator_data.s21.copy()
    
    # Fit raw circle for baseline
    xy = np.column_stack((np.real(resonator_data.s21), np.imag(resonator_data.s21)))
    xc, yc, r = circle_fit_by_taubin(xy)
    resonator_data.raw_circle = {'xc': xc, 'yc': yc, 'r': r}
    
    # Apply corrections in sequence based on options
    if opts["initial_corrects"]:
        resonator_data = _apply_initial_corrections(resonator_data)
    
    if opts["shrink_freq_range"]:
        resonator_data = _shrink_frequency_range(resonator_data)
    
    if opts["phase_slope"]:
        resonator_data = _correct_phase_slope(resonator_data)
    
    if opts["equate_phase_ends"]:
        resonator_data = _equate_phase_ends(resonator_data)
    
    if opts["remove_phase_spiral"]:
        resonator_data = _remove_phase_spiral(resonator_data)
    
    if opts["remove_fit_phase_errors"]:
        resonator_data = _remove_fit_phase_errors(resonator_data, new_modification=False, fit_mode=opts["fit_mode"])
    
    if opts["remove_fit_amplitude_errors"]:
        resonator_data = _remove_fit_amplitude_errors(resonator_data, fit_mode=opts["fit_mode"])
    
    # Calculate SNR metrics
    _calculate_snr_metrics(resonator_data)
    
    return resonator_data

def _apply_initial_corrections(resonator_data, version = 'v2'):
    """Apply initial normalization and phase corrections."""    
    ################# Potential Issues###############
    #       Sensitivity to noise: 
    #       More robust normalization using multiple points from both ends (add after removing phase):
    
    # Get magnitude and phase
    magnitude = np.abs(resonator_data.s21)
    phase = np.unwrap(np.angle(resonator_data.s21))
    
    # Remove mean phase
    phase = phase - np.mean(phase)
    
    if version == 'matlab':
        
        # Perform initial normalization based on first and last data points
        magnitude = magnitude / (0.5 * (magnitude[0] + magnitude[-1]))

    elif version == 'v2':
            #    Using only two points (first and last like the matlab version above) makes the normalization vulnerable to noise in those specific measurements
            #    Assumes baseline stability: Works well only if your baseline is truly flat
            #    Edge effects: If your frequency sweep doesn't extend far enough from resonance, the endpoints may still be affected by the resonance
            #    More robust normalization using multiple points from both ends:
        
        num_points = min(10, len(magnitude) // 10)  # Use 10 points or 10% of data, whichever is smaller  
        baseline = np.mean(np.concatenate([magnitude[:num_points], magnitude[-num_points:]]))
        magnitude = magnitude / baseline

    # Rebuild S21 as the normalized zero mean phase measurement
    resonator_data.s21 = magnitude * np.exp(1j * phase)

    return resonator_data

def _shrink_frequency_range(resonator_data):
    """
    Shrink frequency range to focus on resonance, using ±1000 kHz window around resonance.
    """
    
    # First estimate resonator values to get f0 (resonance frequency)
    estimate = _estimate_resonator_values(resonator_data, method="peak_analysis")
    f0 = estimate['f0']
    
    # Calculate normalized frequency in kHz
    x = (resonator_data.freq - f0) / 1e3
    
    # Find indices closest to ±1000 kHz from resonance
    idx_max = np.argmin(np.abs(x - 1000))
    idx_min = np.argmin(np.abs(x + 1000))
    
    # Store original data
    resonator_data.raw_freq = resonator_data.freq.copy()
    resonator_data.s21_orig = resonator_data.s21.copy()
    
    # Update data with windowed region
    resonator_data.freq = resonator_data.raw_freq[idx_min:idx_max+1]
    resonator_data.s21 = resonator_data.s21_orig[idx_min:idx_max+1]
    
    return resonator_data

def _correct_phase_slope(resonator_data, version = 'v2'):
    """Remove linear phase slope from S21 data."""
    # Extract magnitude and phase
    phase = np.unwrap(np.angle(resonator_data.s21), discont=0.5)
    magnitude = np.abs(resonator_data.s21)
    
    if version == 'matlab':
    # Get endpoints for fitting
        num_points = 10
        x_data = np.concatenate([resonator_data.freq[:num_points], resonator_data.freq[-num_points:]])
        y_data = np.concatenate([phase[:num_points], phase[-num_points:]])
        
        # Fit linear slope to endpoints
        coeffs = np.polyfit(x_data * 1e-9, y_data, 1)
        phase_fit = coeffs[0] * resonator_data.freq * 1e-9 + coeffs[1]
        
        # Remove phase slope and re-center
        corrected_phase = phase - phase_fit
        corrected_phase = corrected_phase - np.mean(corrected_phase)
        
        # Reconstruct S21
        resonator_data.s21 = magnitude * np.exp(1j * corrected_phase)

        return resonator_data

    ############## In matlab this code is used but it doesnt seem to be necessary.  Is this a bug or can we ignore it?
        # S21 = trace.S21;
        # S21 = S21 - (S21(1)+S21(end))/2;
        # trace.S21 = trace.S21;
        
    elif version == 'v2':   
        # # Create mask to exclude resonance region
        # # Identify resonance region (e.g., where magnitude is < mean(magnitude))
        min_idx = np.argmin(magnitude)
        window = len(magnitude) // 5  # 20% of data points
        mask = np.ones(len(magnitude), dtype=bool)
        mask[max(0, min_idx-window):min(len(magnitude), min_idx+window)] = False
        
        # Fit line to masked data
        x = resonator_data.freq * 1e-9  # Scale to GHz for numerical stability
        coeffs = np.polyfit(x[mask], phase[mask], 1)
        phase_fit = coeffs[0] * x + coeffs[1]
        
        # Remove phase slope and re-center
        corrected_phase = phase - phase_fit
        corrected_phase = corrected_phase - np.mean(corrected_phase)
        
        # Reconstruct S21
        resonator_data.s21 = magnitude * np.exp(1j * corrected_phase)
        
        return resonator_data

def _equate_phase_ends(resonator_data):
    """Equate the phase of the start and end points to remove electronic path length effects."""
    num_avg = 5
    
    # Calculate average of first and last points
    min_freq = np.mean(resonator_data.freq[:num_avg])
    max_freq = np.mean(resonator_data.freq[-num_avg:])
    first_s21 = np.mean(resonator_data.s21[:num_avg])
    last_s21 = np.mean(resonator_data.s21[-num_avg:])
    
    # Calculate phase slope
    first_phase = np.angle(first_s21)
    last_phase = np.angle(last_s21)
    
    # Only apply if significant phase difference
    if abs(last_phase - first_phase) > 0.25:
        phase_slope = (last_phase - first_phase) / (max_freq - min_freq)
        epl_correction = phase_slope * (resonator_data.freq - min_freq) + first_phase
        resonator_data.s21 = resonator_data.s21 * np.exp(-1j * epl_correction)
    
    return resonator_data

def _remove_phase_spiral(resonator_data):
    """
    Correct for residual spiral character in S21 data by adjusting 
    the radius of the first and last measured endpoints.
    """
    # Fit circle to the data
    xy = np.column_stack((np.real(resonator_data.s21), np.imag(resonator_data.s21)))
    xc, yc, R = circle_fit_by_taubin(xy)
    
    # Calculate parameters for the first and last points
    num_avg = 2
    first_s21 = np.mean(resonator_data.s21[:num_avg])
    last_s21 = np.mean(resonator_data.s21[-num_avg:])
    min_freq = np.mean(resonator_data.freq[:num_avg])
    max_freq = np.mean(resonator_data.freq[-num_avg:])
    
    # Calculate corrections
    first_r = first_s21 - (xc + 1j*yc)
    first_ref_r = R * (np.cos(np.angle(first_r)) + 1j*np.sin(np.angle(first_r)))
    first_diff = first_r - first_ref_r
    
    last_r = last_s21 - (xc + 1j*yc)
    last_ref_r = R * (np.cos(np.angle(last_r)) + 1j*np.sin(np.angle(last_r)))
    last_diff = last_r - last_ref_r
    
    # Apply correction
    circle_correction = (last_diff - first_diff) / (max_freq - min_freq)
    epl_correction = circle_correction * (resonator_data.freq - min_freq)
    
    try:
        resonator_data.s21 = resonator_data.s21 * np.exp(-1j * epl_correction)
    except Exception as e:
        print("Warning: Quality factor is too low for correction to work")
        resonator_data.good_data_fit = False
        return resonator_data    
    
    # Normalize
    num_pts = 5
    norm_factor = np.mean(np.concatenate([
        np.abs(resonator_data.s21[:num_pts]),
        np.abs(resonator_data.s21[-num_pts:])
    ]))
    resonator_data.s21 = resonator_data.s21 / norm_factor
    
    return resonator_data

def _remove_fit_phase_errors(resonator_data:ResonatorData, new_modification=False, fit_mode='lmfit'):
    """
    Remove phase errors by fitting the S21 data to a resonator model.
    Matches the MATLAB implementation with both standard and new_modification approaches.
    
    Args:
        resonator_data: ResonatorData object with S21 data
        new_modification: If True, uses the simplified circle-bisection approach
        fit_model: Model to use for fitting ('lmfit' or 'scipy'). Scipy model mimics matlab functionality more closesly
    """
    print("\n***************Correcting for phase slope***************")
    if new_modification:
        # Circle-bisection approach (Chris' new modification)
        try:
            # Recalculate the circle fit on the partially corrected data
            xy = np.column_stack((np.real(resonator_data.s21), np.imag(resonator_data.s21)))
            xc, yc, R = circle_fit_by_taubin(xy)
            
            # Find the direction on the circle that bisects the angles of the spectral endpoints
            first_angle = np.angle((xy[0,0] - xc) + 1j*(xy[0,1] - yc))
            last_angle = np.angle((xy[-1,0] - xc) + 1j*(xy[-1,1] - yc))
            target_angle = 0.5 * (first_angle + last_angle)
            
            print(f"first angle: {first_angle*180/np.pi}")
            print(f"last angle: {last_angle*180/np.pi}")
            print(f"target angle: {target_angle*180/np.pi}")
            
            # Apply phase shift to bring this point to the x-intercept
            x = R * np.cos(target_angle) + xc
            y = R * np.sin(target_angle) + yc
            theta_error = np.angle(x + 1j*y)
            
            # Apply phase shift
            resonator_data.s21 = resonator_data.s21 * np.exp(-1j * theta_error)
            
        except Exception as e:
            print(f"Warning: Phase error correction failed: {e}")
    else:
        # Standard nonlinear fitting approach
        try:
            # Get estimate of resonator parameters
            if not resonator_data.estimate:
                estimate = _estimate_resonator_values(resonator_data, method="peak_analysis")
            else:
                estimate = resonator_data.estimate

            # Extract parameters for fitting
            Qio = estimate['Qi']
            Qco = estimate['Qc']
            fo = estimate['f0']
            phi = estimate['phi']
            f3dB = estimate['f3dB']
            
            # Get the frequency and phase data
            X = resonator_data.freq
            Y = signal.medfilt(np.unwrap(np.angle(1/resonator_data.s21)), 5)  # Smooth using a median filter (similar to MATLAB's smooth)
            
            # Define the resonator model for fitting the phase
            def inv_s21_phase_func(x, Qi, Qc, Fo, phi, A, B, C):
                """Model function for the phase of inverse S21"""
            # This replicates the MATLAB's invS21Func and fabs functions
                # invS21Func = A*exp(1i*theta)*(1+(Qi/Qc)*exp(1i*phi)*1/(1+1i*2*Qi*((x-Fo)/Fo)))
                # fabs = angle(C*exp(1i*(A+B*(x-fo)))*invS21Func)
                z = 1 + 1j * 2 * Qi * ((x - Fo) / Fo)
                s21_inv = C * np.exp(1j * (A + B * (x - fo))) * (1 + (Qi / Qc) * np.exp(1j * phi) / z)
                return np.angle(s21_inv)
            
            # Define bounds and initial parameters
            lower_bounds = [1e2, 1e2, fo-100*f3dB, -np.pi/2, -1.5, -10, 0.5]
            upper_bounds = [10*Qio, 10*Qco, fo+100*f3dB, np.pi/2, 1.5, 10, 1.5]
            p0 = [Qio, Qco, fo, phi, 0.0, 0.0, 1.0]
            
            if fit_mode == 'lmfit':
                # Create a residual function for lmfit
                def residual(params, x, data):
                    Qi = params['Qi'].value
                    Qc = params['Qc'].value 
                    Fo = params['Fo'].value
                    phi = params['phi'].value
                    A = params['A'].value
                    B = params['B'].value
                    C = params['C'].value
                    
                    model = inv_s21_phase_func(x, Qi, Qc, Fo, phi, A, B, C)
                    return model - data
                
                # Create Parameters object
                params = lmfit.Parameters()
                params.add('Qi', value=Qio, min=lower_bounds[0], max=upper_bounds[0])
                params.add('Qc', value=Qco, min=lower_bounds[1], max=upper_bounds[1])
                params.add('Fo', value=fo, min=lower_bounds[2], max=upper_bounds[2])
                params.add('phi', value=phi, min=lower_bounds[3], max=upper_bounds[3])
                params.add('A', value=p0[4], min=lower_bounds[4], max=upper_bounds[4])
                params.add('B', value=p0[5], min=lower_bounds[5], max=upper_bounds[5])
                params.add('C', value=p0[6], min=lower_bounds[6], max=upper_bounds[6])
                
                # Perform the fit
                result = lmfit.minimize(residual, params, args=(X, Y), method='least_squares', 
                                       max_nfev=10000)
                
                # Extract the fitted parameters
                A = result.params['A'].value
                B = result.params['B'].value

                resonator_data.estimate['Qi'] = result.params['Qi'].value
                resonator_data.estimate['Qc'] = result.params['Qc'].value
                resonator_data.estimate['Fo'] = result.params['Fo'].value
                resonator_data.estimate['phi'] = result.params['phi'].value
                resonator_data.estimate['theta'] = A
                resonator_data.estimate['B'] = B
                resonator_data.estimate['A'] = result.params['C'].value



                print(f"Value for A+B*(x-fo) = ({A},{B}) [lmfit]")
            elif fit_mode == 'scipy':
                # Perform the fit using scipy's curve_fit
                params, _ = optimize.curve_fit(
                    inv_s21_phase_func, X, Y,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=10000,
                    method='trf'
                )
                
                # Extract the fitted parameters
                A = params[4]
                B = params[5]
                print(f"Value for A+B*(x-fo) = ({A},{B})")
            
            # Apply the phase correction
            phase_correction = A + B * (X - fo)
            resonator_data.s21 = resonator_data.s21 * np.exp(1j * phase_correction)
            
        except Exception as e:
            print(f"Warning: Quality factor is too low for Fit function to work: {e}")
            resonator_data.good_data_fit = False
    
    return resonator_data

def _remove_fit_amplitude_errors(resonator_data: ResonatorData, fit_mode='lmfit'):
    """
    Correct amplitude errors by scaling based on a resonator model fit.
    This implementation matches the MATLAB approach using nonlinear fitting.
    
    Args:
        resonator_data: ResonatorData object with S21 data
        use_lmfit: If True, uses lmfit for parameter optimization instead of scipy.optimize.curve_fit
    """
    print("\n***************Correcting for amplitude*****************")
    
    # Get resonator parameter estimates
    if not resonator_data.estimate:
        estimate = _estimate_resonator_values(resonator_data, method="peak_analysis")
    else:
        estimate = resonator_data.estimate
    
    # Extract parameters for fitting
    Qio = estimate['Qi']
    Qco = estimate['Qc']
    fo = estimate['f0']
    phi = estimate['phi']
    f3dB = estimate['f3dB']
    
    # Prepare data for fitting
    X = resonator_data.freq
    Y = signal.medfilt(np.abs(1/resonator_data.s21), 5)  # Similar to MATLAB's smooth function
    
    # Define the inverse S21 amplitude model function
    def inv_s21_amp_func(x, Qi, Qc, Fo, phi, A):
        """Model function for the amplitude of inverse S21"""
        z = 1 + 1j * 2 * Qi * ((x - Fo) / Fo)
        s21_inv = A * (1 + (Qi / Qc) * np.exp(1j * phi) / z)
        return np.abs(s21_inv)
    
    try:
        # Define bounds and initial parameters similar to MATLAB
        lower_bounds = [1e3, 1e3, fo-f3dB, -np.pi/2, 0]
        upper_bounds = [10*Qio, 10*Qco, fo+f3dB, np.pi/2, 10]
        p0 = [Qio, Qco, fo, phi, 1.0]
        
        if fit_mode == 'lmfit':
            # Create residual function for lmfit
            def residual(params, x, data):
                Qi = params['Qi'].value
                Qc = params['Qc'].value 
                Fo = params['Fo'].value
                phi = params['phi'].value
                A = params['A'].value
                
                model = inv_s21_amp_func(x, Qi, Qc, Fo, phi, A)
                return model - data
            
            # Create Parameters object
            params = lmfit.Parameters()
            params.add('Qi', value=Qio, min=lower_bounds[0], max=upper_bounds[0])
            params.add('Qc', value=Qco, min=lower_bounds[1], max=upper_bounds[1])
            params.add('Fo', value=fo, min=lower_bounds[2], max=upper_bounds[2])
            params.add('phi', value=phi, min=lower_bounds[3], max=upper_bounds[3])
            params.add('A', value=p0[4], min=lower_bounds[4], max=upper_bounds[4])
            
            # Perform the fit
            result = lmfit.minimize(residual, params, args=(X, Y), method='least_squares',
                                  max_nfev=1000)
            
            # Extract the amplitude scaling factor
            A = result.params['A'].value
            # resonator_data.estimate['Qi'] = result.params['Qi'].value
            # resonator_data.estimate['Qc'] = result.params['Qc'].value
            # resonator_data.estimate['Fo'] = result.params['Fo'].value
            # resonator_data.estimate['phi'] = result.params['phi'].value
            # resonator_data.estimate['A'] = A
            # resonator_data.estimate['B'] = result.params['B'].value
            # resonator_data.estimate['C'] = result.params['C'].value
            print(f"Value for A = ({A}) [lmfit]")
        elif fit_mode == 'scipy':
            # Perform nonlinear fitting with scipy
            params, _ = optimize.curve_fit(
                inv_s21_amp_func, X, Y,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=1000,
                method='trf'
            )
            
            # Extract the amplitude scaling factor
            A = params[4]
            print(f"Value for A = ({A})")
        
        # Apply amplitude scaling if within reasonable range
        if 0.5 < A < 2:
            # The fitting is completed on iS21, so multiplying by A is correct
            resonator_data.s21 = resonator_data.s21 * A
        else:
            print("Warning: A<0.5 or A>2, Amplitude fix not applied")
        
    except Exception as e:
        print(f"Warning: Quality factor is too low for Fit function to work: {e}")
        resonator_data.good_data_fit = False
    
    return resonator_data

def _calculate_snr_metrics(resonator_data):
    """Calculate signal-to-noise ratio metrics for the resonator data."""
    # Calculate SNR metrics
    resonator_data.s21mag_snr = calc_mag_snr(resonator_data.s21)
    resonator_data.s21circ_snr = calc_circ_snr(resonator_data.s21, True)
    
    # Also calculate for inverse S21
    resonator_data.is21 = 1.0 / resonator_data.s21
    resonator_data.is21mag_snr = calc_mag_snr(resonator_data.is21)
    resonator_data.is21circ_snr = calc_circ_snr(resonator_data.is21, False)
    
    return resonator_data

def _estimate_resonator_values(resonator_data, method="peak_analysis"):
    """
    Estimate resonator parameters based on S21 data.
    
    Args:
        resonator_data: ResonatorData object with S21 data
        method: Method to use for estimation ("manual" or "peak_analysis")
        
    Returns:
        Dictionary of estimated values
    """
    if method == "peak_analysis":
        # Use peak_analysis module to estimate resonator parameters
        magnitude = np.abs(resonator_data.s21)
        
        # Find resonance using peak_info (treating resonance as a dip)
        peaks_data = peak_info(
            data=magnitude,
            frequencies=resonator_data.freq,
            expectedPeaks=1,  # We expect one resonance
            is_dip=True,      # Resonance is a dip in |S21|
            rel_height=0.5    # Half-maximum for width calculation
        )
        
        # Extract resonance parameters from peaks_data
        if len(peaks_data['peakIndex']) > 0:
            # We found a resonance
            idx = 0  # First (and only) peak
            f0 = peaks_data['fo'][idx]
            f3dB = peaks_data['fwhm_freq'][idx]
            qt = peaks_data['q'][idx]
            
            # Get circle parameters
            xy = np.column_stack((np.real(resonator_data.s21), np.imag(resonator_data.s21)))
            xc, yc, r = circle_fit_by_taubin(xy)
            resonator_data.ref_circle = {'xc': xc, 'yc': yc, 'r': r}
            
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
        else:
            # No resonance found - fall back to manual method
            print("Warning: No resonance found with peak_analysis, falling back to manual method")
            return _estimate_resonator_values(resonator_data, method="manual")
            
    elif method == "manual":
        # Original "manual" implementation
        # Get magnitude and phase
        magnitude = np.abs(resonator_data.s21)
        phase = np.unwrap(np.angle(resonator_data.s21))
        
        # Find resonance frequency (minimum of |S21|)
        min_idx = np.argmin(magnitude)
        f0 = resonator_data.freq[min_idx]
        
        # Estimate FWHM
        half_max = (np.max(magnitude) + np.min(magnitude)) / 2
        idx_above = magnitude > half_max
        
        # Find left and right crossing points
        left_idx = np.where(idx_above[:min_idx])[0]
        right_idx = np.where(idx_above[min_idx:])[0] + min_idx
        
        # Calculate f3dB (full width at half maximum)
        if len(left_idx) > 0 and len(right_idx) > 0:
            f_left = resonator_data.freq[left_idx[-1]]
            f_right = resonator_data.freq[right_idx[0]]
            f3dB = f_right - f_left
        else:
            f3dB = (resonator_data.freq[-1] - resonator_data.freq[0]) / 10
        
        # Calculate quality factors
        qt = f0 / f3dB
        
        # Get circle parameters
        xy = np.column_stack((np.real(resonator_data.s21), np.imag(resonator_data.s21)))
        xc, yc, r = circle_fit_by_taubin(xy)
        resonator_data.ref_circle = {'xc': xc, 'yc': yc, 'r': r}
        
        # Estimate coupling phase using circle fit
        phi = np.angle(complex(xc, yc))
        
        # Estimate internal and coupling Q
        S_min = np.min(magnitude)
        S_max = np.max(magnitude)
        
        # Simple estimate using diameter method
        coupling_factor = (S_max - S_min) / (S_max + S_min)
        if coupling_factor < 1:  # Undercoupled
            qc = qt / coupling_factor
            qi = qt / (1 - coupling_factor)
        else:  # Overcoupled
            qc = qt
            qi = qt * 2
    else:
        raise ValueError(f"Unknown method '{method}'. Must be 'manual' or 'peak_analysis'")
    
    # Store estimates in a dictionary
    estimate = {
        'f0': f0,
        'f3dB': f3dB,
        'Qt': qt,
        'Qi': qi,
        'Qc': qc,
        'phi': phi,
    }
    
    resonator_data.estimate = estimate
    return estimate