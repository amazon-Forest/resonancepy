"""
Resonator fitting module for the ResonatorProject package.
This module implements the equivalent of SingleResonatorFit.m from MATLAB.

The module provides functionality for:
1. Fitting microwave resonator data in the form of S21 scattering parameters
2. Calculating quality factors (Qi, Qc, Qt) and resonant frequencies
3. Producing standardized plots and analyses reports
4. Converting between S21 and inverse S21 representations for fitting optimization
5. Saving results to HDF5 files with metadata

The fitting is based on the circle-fitting approach described in:
- Probst et al., "Efficient and robust analysis of complex scattering data
  under noise in microwave resonators", Rev. Sci. Instrum. 86, 024706 (2015)
- Khalil et al., "An analysis method for asymmetric resonator transmission applied to 
  superconducting devices", J. Appl. Phys. 111, 054510 (2012)
"""
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import optimize
from typing import Dict, Any, Optional, Tuple, Union, List
from lmfit import Model, Parameters, minimize
import io
import h5py
from PIL import Image
import warnings


from resonance_py.data_analysis.resonator_data import ResonatorData
from resonance_py.data_analysis.artifacts import fix_artifacts
from resonance_py.data_analysis.snr_metrics import calc_mag_snr, calc_circ_snr
from resonance_py.data_analysis.circle_fitting import circle_fit_by_taubin


def single_resonator_fit(resonator_data: ResonatorData = None, opts: Dict[str, Any] = None) -> ResonatorData:
    """
    Main function for fitting superconducting resonator data using the inverse S21 method.
    This is the Python equivalent of MATLAB's SingleResonatorFit.m.
    
    The function performs several steps:
    1. Handles input data and default options
    2. Fixes artifacts in the S21 data
    3. Performs the inverse S21 fitting routine
    4. Calculates SNR metrics and other quality indicators
    5. Calculates photon number and power metrics
    6. Consolidates results into a standardized format
    7. Plots and optionally saves the results
    
    Args:
        resonator_data: ResonatorData object containing:
            - freq: Frequency points array
            - s21: Complex S21 transmission data array
            - is21: Complex inverse S21 data (1/s21)
            - estimate: Initial parameter estimates dictionary
            - Other metadata (temperature, power, etc.)
        
        opts: Dictionary containing options with possible keys:
            - fix_artifacts_opts: Options for artifact fixing algorithm
            - fit_opts: Options for the fitting procedure
            - analysis_foldertag: Additional folder tag for saving results
            
    Returns:
        ResonatorData object with added fit results and metrics:
            - fit: Dictionary containing all fitting parameters and errors
            - SNR metrics for both S21 and inverse S21
            - Power and photon number calculations
            - Consolidated text summary of results
    """
    # Handle default arguments
    if resonator_data is None:
        # Load data if not provided (equivalent to ResData_LoadS21 in MATLAB)
        # Note: This functionality needs to be implemented separately
        resonator_data = load_resonator_data()
        opts = {"fix_artifacts_opts": {}, "fit_opts": {}, "plot": False}
    elif opts is None:
        opts = {"fix_artifacts_opts": {}, "fit_opts": {}, "plot": False}
    else:
        if "fix_artifacts_opts" not in opts:
            opts["fix_artifacts_opts"] = {}
        if "fit_opts" not in opts:
            opts["fit_opts"] = {}
        if "plot" not in opts:
            opts["plot"] = False

    # Settings equivalent to MATLAB's DR-D settings
    if not hasattr(resonator_data, "system_attenuation"):
        resonator_data.system_attenuation = 70  # Default attenuation in dB
        resonator_data.launch_power = 0  # Default launch power in dBm

    # Set up filenames for saving results
    if hasattr(resonator_data, "base_filename"):
        resonator_data.s21_graph_filename = f"{resonator_data.base_filename}.jpg"

    # Fix artifacts in the data (e.g., phase unwrapping issues, noise reduction)
    resonator_data = fix_artifacts(resonator_data, opts["fix_artifacts_opts"])

    # Apply the fit using inverse S21 approach (better for high-Q resonators)
    if resonator_data.fit["model"] == "Probst":
        print("Using Probst fitting model")
        resonator_data = probst_fit(resonator_data, opts["fit_opts"])
    else:
        resonator_data = res_is21_fit(resonator_data, opts["fit_opts"])
    

    # Calculate Signal-to-Noise Ratio metrics for both S21 and inverse S21
    # Higher SNR indicates better data quality for fitting
    resonator_data.s21mag_snr = calc_mag_snr(resonator_data.s21)  # Magnitude SNR for S21
    resonator_data.s21circ_snr = calc_circ_snr(resonator_data.s21, True)  # Circle SNR for S21
    resonator_data.is21mag_snr = calc_mag_snr(resonator_data.is21)  # Magnitude SNR for inverse S21
    resonator_data.is21circ_snr = calc_circ_snr(resonator_data.is21, False)  # Circle SNR for inverse S21

    # Calculate magnitude in dB for reporting purposes
    resonator_data.mag_s21_postfit = 20 * np.log10(np.abs(resonator_data.s21))  # in dBm

    # Calculate min and max S21 using moving average to reduce noise influence
    window_size = 5  # Points for smoothing, reduces influence of outliers
    resonator_data.min_s21 = np.min(_moving_average(resonator_data.mag_s21_postfit, window_size))
    resonator_data.max_s21 = np.max(_moving_average(resonator_data.mag_s21_postfit, window_size))

    # Define save paths and create directories for storing results
    if hasattr(resonator_data, "data_pathname") and hasattr(resonator_data, "save_path_nametag"):
        today = datetime.datetime.today().strftime('%Y%m%d')
        save_path_nametag = resonator_data.save_path_nametag

        # Handle override flag for overwriting existing files
        if hasattr(resonator_data, "override_flag") and resonator_data.override_flag:
            save_path_nametag = f"{save_path_nametag}override-"

        # Create the main save path with date stamp
        resonator_data.save_pathname = os.path.join(
            resonator_data.data_pathname, 
            f"{save_path_nametag}{today}"
        )

        # Create additional subdirectory if specified (for categorizing analyses)
        if "analysis_foldertag" in opts:
            resonator_data.save_pathname = os.path.join(
                resonator_data.save_pathname,
                opts["analysis_foldertag"]
            )

        # Ensure directory exists
        os.makedirs(resonator_data.save_pathname, exist_ok=True)

    # Calculate photon number in the resonator and related power metrics
    # These calculations follow standard methods in circuit QED literature
    resonator_data.total_atten = resonator_data.system_attenuation + resonator_data.atten  # Total attenuation in dB
    # Convert from dBm to Watts at the device, accounting for attenuation
    resonator_data.dut_power = 1e-3 * 10**(resonator_data.launch_power/10) * 10**(-resonator_data.total_atten/10)
    # Calculate photon lifetime from total quality factor
    resonator_data.lifetime = resonator_data.fit["Qt"] / (np.pi * resonator_data.fit["fo"])

    # Calculate circulating power and photon number following Bruno APL method
    # Reference: Bruno et al., Appl. Phys. Lett. 106, 182601 (2015)
    h = 6.626069e-34  # Planck constant (J·s)
    # Circulating power calculation using Q factors
    resonator_data.circulating_power = resonator_data.dut_power * 2 * resonator_data.fit["Qt"]**2 / (
        resonator_data.fit["Qc"] * resonator_data.fit["Qi"]
    )
    # Average photon number in steady state (n_photon = P_circ/(hf*rate))
    resonator_data.Np = (resonator_data.dut_power * resonator_data.fit["Qt"]**2) / (
        resonator_data.fit["Qc"] * h * np.pi * resonator_data.fit["fo"]**2
    )
    # trace.Np = (trace.dutPower*trace.fit.Qt^2)/(trace.fit.Qc*h*pi*trace.fit.fo^2);
    # Consolidate results into a human-readable summary
    _consolidate_results(resonator_data)

    # Plot and save data visualization
    if opts["plot"]:
        resonator_fit_plot(resonator_data)

    # Display results in the console
    for line in resonator_data.fit["spectral_fit_results_summary"]:
        print(line)

    return resonator_data

# Helper function for moving average
def _moving_average(data, window_size):
    """
    Calculate moving average of data with given window size.
    
    This function applies a simple box filter to smooth the data and 
    reduce the influence of outliers or noise.
    
    Args:
        data: 1D numpy array containing the data to be smoothed
        window_size: Integer size of the moving average window
        
    Returns:
        1D numpy array of smoothed data with same size as input
    
    Note:
        Uses 'same' mode for convolution to maintain the same array size
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def res_is21_fit(resonator_data: ResonatorData, fit_opts: Dict[str, Any] = None) -> ResonatorData:
    """
    Fits the inverse S21 data to extract resonator parameters.
    
    This function implements the main fitting algorithm for superconducting resonators,
    using the inverse S21 representation which provides better fitting stability for 
    high-Q resonators. The function uses a physics-based model that includes:
    - Internal quality factor (Qi)
    - Coupling quality factor (Qc)
    - Resonance frequency (fo)
    - Coupling phase (phi)
    - Global rotation angle (theta)
    - Amplitude scaling (A)
    - Linear background term (B)
    
    The fitting procedure uses the lmfit package to perform non-linear least squares 
    optimization of the model parameters.
    
    Args:
        resonator_data: ResonatorData object with:
            - freq: Frequency points array in Hz
            - s21: Complex S21 transmission data
            - is21: Complex inverse S21 data (1/s21)
            - estimate: Initial parameter estimates dictionary
        
        fit_opts: Dictionary of fitting options including:
            - debug_plotting: Boolean to enable/disable debug plots
            - fixed_params: Dictionary of parameters to hold fixed during fitting
        
    Returns:
        ResonatorData object with added fit dictionary containing:
            - Qi, Qc, Qt: Quality factors
            - fo: Resonance frequency (Hz)
            - phi, theta: Phase angles (radians)
            - A, B: Amplitude scaling and background parameters
            - Error estimates for all parameters
            - Model predictions for s21 and is21
            - Goodness-of-fit metrics
    
    References:
        - Khalil et al., J. Appl. Phys. 111, 054510 (2012)
        - Probst et al., Rev. Sci. Instrum. 86, 024706 (2015)
    """
    # Default fit options
    opts = {
        "debug_plotting": False,  # Enable for debug plots during fitting process
        "fixed_params": {}        # Parameters to hold fixed during optimization
    }
    
    # Update with user options
    if fit_opts:
        opts.update(fit_opts)
    
    # Get initial estimates from the data
    init_estimates = resonator_data.estimate
    
    # Define the fitting model for complex S21^-1 with scaled parameters
    def inv_s21_model(f, Qi_scaled, Qc_scaled, fo_scaled, phi, theta, A, B):
        """
        Physics-based model for inverse S21 with scaled parameters.
        
        This model implements the asymmetric resonator model described by Khalil et al.
        with modifications for practical fitting. The parameters are scaled to improve
        numerical stability during optimization.
        
        The model includes:
        - Resonator intrinsic response (Qi, Qc, fo)
        - Phase shift from coupling (phi)
        - Global rotation of the entire S21 circle (theta)
        - Amplitude scaling factor (A)
        - Linear background term for phase slope (B)
        
        Args:
            f: Frequency points (Hz)
            Qi_scaled: Internal quality factor (scaled by 1e-6)
            Qc_scaled: Coupling quality factor (scaled by 1e-6)
            fo_scaled: Resonance frequency (scaled by 1e-9 - GHz)
            phi: Coupling phase (radians)
            theta: Global rotation angle (radians)
            A: Amplitude scale factor
            B: Linear background (scaled by 1e-3)
            
        Returns:
            Complex inverse S21 values
            
        References:
            Khalil et al., J. Appl. Phys. 111, 054510 (2012)
        """
        # Scale parameters back to their natural units
        Qi = Qi_scaled * 1e6    # Internal Q (typical range: 10^4 - 10^7)
        Qc = Qc_scaled * 1e6    # Coupling Q (typical range: 10^4 - 10^6)
        fo = fo_scaled * 1e9    # Resonance frequency (typical range: 1-10 GHz)
        
        # Calculate the normalized frequency detuning
        deltax = (f - fo) / fo  # Normalized detuning

        # Real and imaginary parts without global phase rotation
        # These equations implement the physical model from Khalil et al.
        # Fix: Use consistent scaling
        f1 = 1 + (Qi/Qc) * (np.cos(phi) + 2*Qi*deltax*np.sin(phi)) / (1+(2*Qi*deltax)**2)
        f2 = 0 + (Qi/Qc) * (np.sin(phi) - 2*Qi*deltax*np.cos(phi)) / (1+(2*Qi*deltax)**2)
    
        # Apply amplitude scaling correction
        f1 = A * f1  # Scale real part
        f2 = A * f2  # Scale imaginary part
        
        # Apply global rotation and linear phase correction
        B_scaled = B * 1e3  # Scale back the background term
        phase_shift = theta - B_scaled * deltax
        real_part = f1 * np.cos(phase_shift) - f2 * np.sin(phase_shift)
        imag_part = f2 * np.cos(phase_shift) + f1 * np.sin(phase_shift)
        
        # Return complex result
        return real_part + 1j * imag_part
    
    # Define parameters for optimization with appropriate bounds and scaling
    params = Parameters()
    # Internal Q with appropriate bounds for superconducting resonators
    params.add('Qi_scaled', value=init_estimates["Qi"]/1e6, min=0.1, max=100)
    # Coupling Q with bounds based on typical experimental values
    params.add('Qc_scaled', value=init_estimates["Qc"]/1e6, min=0.0, max=0.5)
    # Resonance frequency, limited to ±1% of initial estimate
    params.add('fo_scaled', value=init_estimates["f0"]/1e9, 
               min=init_estimates["f0"]/1e9*0.99, max=init_estimates["f0"]/1e9*1.01)
    # Coupling phase limited to physical range
    params.add('phi', value=init_estimates["phi"], min=-np.pi/2, max=np.pi/2)
    # Global rotation angle
    params.add('theta', value=init_estimates["theta"], min=-np.pi/2, max=np.pi/2)
    # Amplitude scaling close to unity
    params.add('A', value=init_estimates["A"], min=0.9, max=1.1)
    # Linear background term for phase slope correction
    params.add('B', value=init_estimates["B"], min=-10, max=10)
    
    # Apply any user-specified fixed parameters
    for param_name, value in opts["fixed_params"].items():
        if param_name in params:
            params[param_name].set(value=value, vary=False)
        
    # Define residual function for optimization
    def residual(params, freqs, data):
        """
        Calculate the residual between model and data for optimization.
        
        This function extracts parameter values from the lmfit Parameters object,
        calculates the model prediction, and returns the difference between model
        and data as a flattened array of real values (for both real and imaginary parts).
        
        Args:
            params: lmfit Parameters object containing model parameters
            freqs: Frequency points array (Hz)
            data: Complex S21 data to fit
            
        Returns:
            Flattened array of residuals (real followed by imaginary)
        """
        # Extract parameters from the Parameters object
        Qi_scaled = params['Qi_scaled'].value
        Qc_scaled = params['Qc_scaled'].value
        fo_scaled = params['fo_scaled'].value
        phi = params['phi'].value
        theta = params['theta'].value
        A = params['A'].value
        B = params['B'].value

        # Calculate model prediction
        model = inv_s21_model(freqs, Qi_scaled, Qc_scaled, fo_scaled, phi, theta, A, B)
    
        # Return complex residuals as flattened real-valued array
        # This is necessary because optimization algorithms work with real values
        return np.concatenate([np.real(data - model), np.imag(data - model)])
    
    
    # Perform the fit
    try:
        # Perform the optimization using least squares method
        result = minimize(residual, params, args=(resonator_data.freq, resonator_data.is21),
                          method='least_squares')  # Levenberg-Marquardt algorithm
        
        # Extract fitted parameters
        # Access individual parameters from the result
        Qi_scaled = result.params['Qi_scaled'].value
        Qc_scaled = result.params['Qc_scaled'].value
        fo_scaled = result.params['fo_scaled'].value
        phi = result.params['phi'].value
        theta = result.params['theta'].value
        A = result.params['A'].value
        B = result.params['B'].value
        
        # Scale back to natural units for reporting and analysis
        Qi = Qi_scaled * 1e6  # Internal quality factor
        Qc = Qc_scaled * 1e6  # Coupling quality factor 
        fo = fo_scaled * 1e9  # Resonance frequency in Hz
        B = B * 1e-3         # Background term scaled back
        
        # Calculate total quality factor from internal and coupling Q
        # Qt = (Qi*Qc)/(Qi+Qc) - standard formula from parallel circuit analogy
        Qt = (Qi * Qc) / (Qi + Qc)
        
        # Calculate model predictions for both inverse S21 and S21
        model_is21 = inv_s21_model(resonator_data.freq, Qi_scaled, Qc_scaled, fo_scaled, phi, theta, A, B)
        model_s21 = 1 / model_is21  # Convert inverse S21 to S21

        # (Optionally) replicate final circle shift from MATLAB, if you do that
        # deltax = (resonator_data.freq - fo)/fo
        # model_is21 = model_is21 * np.exp(-1j*((-theta) + (B*1e3)*deltax))
        # model_s21  = 1.0 / model_is21
        
        # Calculate goodness of fit metrics for reporting
        # Magnitude fits
        S21_mag = np.abs(resonator_data.s21)
        S21_phase = np.unwrap(np.angle(resonator_data.s21))
        iS21_mag = np.abs(resonator_data.is21)
        iS21_phase = np.unwrap(np.angle(resonator_data.is21))
        
        model_S21_mag = np.abs(model_s21)
        model_S21_phase = np.unwrap(np.angle(model_s21))
        model_iS21_mag = np.abs(model_is21)
        model_iS21_phase = np.unwrap(np.angle(model_is21))
        
        # Calculate root mean square error (RMSE) for magnitude and phase
        s21mag_fit_rmse = np.sqrt(np.mean((S21_mag - model_S21_mag)**2))
        s21phase_fit_rmse = np.sqrt(np.mean((S21_phase - model_S21_phase)**2))
        is21mag_fit_rmse = np.sqrt(np.mean((iS21_mag - model_iS21_mag)**2))
        is21phase_fit_rmse = np.sqrt(np.mean((iS21_phase - model_iS21_phase)**2))
        
        # Calculate circle fit RMSE (Euclidean distance in complex plane)
        xy_s21 = np.column_stack((np.real(resonator_data.s21), np.imag(resonator_data.s21)))
        xy_is21 = np.column_stack((np.real(resonator_data.is21), np.imag(resonator_data.is21)))
        xy_model_s21 = np.column_stack((np.real(model_s21), np.imag(model_s21)))
        xy_model_is21 = np.column_stack((np.real(model_is21), np.imag(model_is21)))
        
        s21circle_fit_rmse = np.sqrt(np.mean(np.sum((xy_s21 - xy_model_s21)**2, axis=1)))
        is21circle_fit_rmse = np.sqrt(np.mean(np.sum((xy_is21 - xy_model_is21)**2, axis=1)))
        
        # Calculate confidence intervals from covariance matrix
        try:
            # Get the covariance matrix - approach based on standard statistical methods
            N = len(resonator_data.freq) * 2  # Number of data points (real + imag)
            p = len(result.params)  # Number of parameters
            
            # Calculate parameter standard errors from covariance matrix
            # This uses standard statistical formulas for non-linear least squares
            pcov = result.covar  # Get covariance matrix directly from result
            perr = np.sqrt(np.diag(pcov))  # Standard errors are sqrt of diagonal
            
            # Scale by 1.96 for ~95% confidence interval (assumes normal distribution)
            QiError = perr[0] * 1.96 * 1e6  # Scale back up for internal Q
            QcError = perr[1] * 1.96 * 1e6  # Scale back up for coupling Q
            foError = perr[2] * 1.96 * 1e9  # Scale back up for frequency
            phiError = perr[3] * 1.96       # Phase error
            thetaError = perr[4] * 1.96     # Rotation angle error
            AError = perr[5] * 1.96         # Amplitude scaling error
            BError = perr[6] * 1.96 * 1e-3  # Background error (scaled)
            
            # Calculate QtError using error propagation formula
            # For Qt = (Qi*Qc)/(Qi+Qc), apply standard error propagation
            QtError = Qt * np.sqrt((QiError/Qi)**2 + (QcError/Qc)**2)
            
        except Exception as e:
            # Fallback error estimates if covariance calculation fails
            # These provide reasonable error estimates based on typical uncertainty
            QiError = Qi * 0.1     # 10% error estimate for Qi
            QcError = Qc * 0.1     # 10% error estimate for Qc
            QtError = Qt * 0.1     # 10% error estimate for Qt
            foError = fo * 1e-4    # 0.01% error for frequency
            phiError = phi * 0.1   # 10% error for coupling phase
            thetaError = theta * 0.1  # 10% error for rotation angle
            AError = A * 0.1       # 10% error for amplitude
            BError = B * 0.1       # 10% error for background
        
        # Store fit results in the resonator_data object
        fit_results = {
            # Main resonator parameters
            "Qi": Qi,              # Internal quality factor
            "Qc": Qc,              # Coupling quality factor
            "Qt": Qt,              # Total quality factor
            "fo": fo,              # Resonance frequency (Hz)
            
            # Additional model parameters
            "phi": phi,            # Coupling phase (radians)
            "theta": theta,        # Global rotation angle (radians)
            "A": A,                # Amplitude scaling factor
            "B": B,                # Linear background term
            
            # Parameter uncertainties (95% confidence intervals)
            "QiError": QiError,    # Internal Q error
            "QcError": QcError,    # Coupling Q error
            "QtError": QtError,    # Total Q error
            "foError": foError,    # Frequency error (Hz)
            "phiError": phiError,  # Coupling phase error
            "thetaError": thetaError,  # Rotation angle error
            "AError": AError,      # Amplitude scaling error
            "BError": BError,      # Background term error
            
            # Model predictions for plotting and analysis
            "model_s21": model_s21,     # Model prediction for S21
            "model_is21": model_is21,   # Model prediction for inverse S21
            
            # Goodness of fit metrics
            "gof": {
                "S21magFitRMSE": s21mag_fit_rmse,      # Magnitude RMSE for S21
                "S21PhaseFitRMSE": s21phase_fit_rmse,  # Phase RMSE for S21
                "iS21magFitRMSE": is21mag_fit_rmse,    # Magnitude RMSE for inverse S21
                "iS21PhaseFitRMSE": is21phase_fit_rmse,  # Phase RMSE for inverse S21
                "S21CircleFitRMSE": s21circle_fit_rmse,  # Complex plane RMSE for S21
                "iS21CircleFitRMSE": is21circle_fit_rmse,  # Complex plane RMSE for inverse S21
            }
        }

        if not hasattr(resonator_data, 'fit'):
            resonator_data.fit = {}

        # Update with new results
        resonator_data.fit.update(fit_results)

    
    except Exception as e:
        # Handle fitting failures gracefully
        print(f"Error in fitting: {e}")
        # Set default values in case of fitting failure, using initial estimates
        resonator_data.fit = {
            # Use initial estimates when fitting fails
            "Qi": init_estimates["Qi"],
            "Qc": init_estimates["Qc"],
            "Qt": init_estimates["Qt"],
            "fo": init_estimates["f0"],
            "phi": init_estimates["phi"],
            "theta": 0.0,
            "A": 1.0,
            "B": 0.0,
            
            # Large error bars to indicate low confidence
            "QiError": init_estimates["Qi"] * 0.5,
            "QcError": init_estimates["Qc"] * 0.5,
            "QtError": init_estimates["Qt"] * 0.5,
            "foError": init_estimates["f0"] * 1e-4,
            "phiError": 0.5,
            "thetaError": 0.5,
            "AError": 0.5,
            "BError": 0.1,
            
            # Use original data as "model" since fitting failed
            "model_s21": resonator_data.s21,
            "model_is21": resonator_data.is21,
            
            # Set standard goodness-of-fit metrics to indicate poor fit
            "gof": {
                "S21magFitRMSE": 1.0,
                "S21PhaseFitRMSE": 1.0,
                "iS21magFitRMSE": 1.0,
                "iS21PhaseFitRMSE": 1.0,
                "S21CircleFitRMSE": 1.0,
                "iS21CircleFitRMSE": 1.0,
            }
        }
    
    return resonator_data

def probst_fit(resonator_data: ResonatorData, initial_guess: Dict[str, Any] = None) -> ResonatorData:

    def complex_s21_model_qi_qc(f, fr, Qi, Qc_mag, phi, A, theta, B):
        """
        Complex S21 model, parameterized by Qi and Qc directly.
        Background uses A (amplitude), theta (global phase), B (linear phase slope vs detuning).

        Args:
            f (array): Frequency points (Hz).
            fr (float): Resonance frequency (Hz).
            Qi (float): Internal quality factor.
            Qc_mag (float): Absolute value (magnitude) of the coupling quality factor.
            phi (float): Coupling phase (of Qc = Qc_mag * exp(-i*phi)) (radians).
            A (float): Background transmission amplitude scaling.
            theta (float): Background global phase offset (radians).
            B (float): Background linear phase slope vs detuning (dimensionless).

        Returns:
            array: Complex S21 values predicted by the model.
        """
        f = np.array(f, dtype=float)

        # --- Calculate Ql from Qi and Qc ---
        # Re{1/Qc} = Re{1 / (Qc_mag * exp(-i*phi))} = cos(phi) / Qc_mag
        re_inv_qc = np.cos(phi) / Qc_mag
        inv_Ql = (1.0 / Qi) + re_inv_qc

        # Ensure Ql is positive; handle potential numerical issues or invalid inputs
        if inv_Ql <= 1e-18: # Use a small threshold instead of zero
            # Return NaN or a large number to signal fit error, or use a floor
            # Forcing a floor might be okay if bounds are set reasonably.
            warnings.warn(f"Calculated 1/Ql = {inv_Ql:.2e} <= 0. Check parameters (Qi={Qi:.2e}, Qc_mag={Qc_mag:.2e}, phi={phi:.3f}). Clamping Ql.", RuntimeWarning)
            Ql = 1e12 # Assign a very large Ql if inv_Ql is non-positive
        else:
            Ql = 1.0 / inv_Ql
        # --- End Ql Calculation ---

        Qc_complex = Qc_mag * np.exp(-1j * phi)

        # Dimensionless detuning
        x = (f - fr) / fr

        # Resonator part (same structure as before, using calculated Ql)
        denom = 1.0 + 2.0j * Ql * x
        resonator_term = (Ql / Qc_complex) / denom

        # --- Background Calculation with A, theta, B ---
        # Phase = theta + B*x  (Linear phase vs detuning)
        bkg_phase = theta + B * x
        background = A * np.exp(1j * bkg_phase)
        # --- End Background Calculation ---

        return background * (1.0 - resonator_term)

    def residual_circlefit_qi_qc(params, f, data_real, data_imag):
        """
        Residual function for the Qi/Qc parameterized model.
        """
        fr = params['fr'].value
        Qi = params['Qi'].value
        Qc_mag = params['Qc_mag'].value
        phi = params['phi'].value
        A = params['A'].value
        theta = params['theta'].value
        B = params['B'].value

        # Evaluate model
        model_complex = complex_s21_model_qi_qc(f, fr, Qi, Qc_mag, phi, A, theta, B)

        # Handle potential NaN from model if parameters were bad
        if np.any(np.isnan(model_complex)):
            return np.full(len(f)*2, 1e6) # Return large residual if model failed

        model_real = np.real(model_complex)
        model_imag = np.imag(model_complex)

        # Return combined residual: (data - model)
        resid = np.concatenate([(data_real - model_real),
                                (data_imag - model_imag)])
        return resid

    def fit_resonator_circle_qi_qc(frequency, S21_complex, init_guess=None, fit_background=True):
        """
        Perform the circle fit using the Qi/Qc parameterized model.

        frequency:      np array of frequency points [Hz]
        S21_complex:    np array of complex S21 data measured
        init_guess:     dict or None, optional initial guesses for Qi, Qc_mag, fr, etc.
        fit_background: bool, decide whether to fit for A, theta, B

        returns: lmfit MinimizerResult and the best-fit model trace
        """
        data_real = np.real(S21_complex)
        data_imag = np.imag(S21_complex)

        params = Parameters()

        # --- Better Initial Guesses ---
        # Use data to make smarter guesses
        idx_min = np.argmin(np.abs(S21_complex))
        fr_guess = frequency[idx_min]

        mag_db = 20 * np.log10(np.abs(S21_complex))
        min_mag_db = np.min(mag_db)
        # Estimate baseline away from resonance (e.g., outer 10% of points)
        edge_indices = int(0.1 * len(frequency))
        baseline_mag_db = np.mean(np.concatenate((mag_db[:edge_indices], mag_db[-edge_indices:])))
        half_max_db = (baseline_mag_db + min_mag_db) / 2

        try:
            left_idx = np.where(mag_db[:idx_min] > half_max_db)[0][-1]
            right_idx = np.where(mag_db[idx_min:] > half_max_db)[0][0] + idx_min
            fwhm = frequency[right_idx] - frequency[left_idx]
            Ql_guess_est = fr_guess / fwhm # Estimate Ql first
        except IndexError:
            fwhm = (frequency[-1] - frequency[0]) / 10
            Ql_guess_est = fr_guess / fwhm

        # Rough guess relations (assuming near critical coupling might be okay start)
        Qi_guess = 2 * Ql_guess_est
        Qc_mag_guess = 2 * Ql_guess_est
        phi_guess = 0.0

        # Background guesses
        baseline_A_guess = np.mean(np.abs(np.concatenate((S21_complex[:edge_indices], S21_complex[-edge_indices:]))))
        baseline_phase = np.unwrap(np.angle(S21_complex))
        # Estimate theta and B from baseline phase slope vs detuning
        edge_detuning = (np.concatenate((frequency[:edge_indices], frequency[-edge_indices:])) - fr_guess) / fr_guess
        edge_phase = np.concatenate((baseline_phase[:edge_indices], baseline_phase[-edge_indices:]))
        try:
            poly_coeffs = np.polyfit(edge_detuning, edge_phase, 1)
            B_guess = poly_coeffs[0]
            theta_guess = poly_coeffs[1] # Phase at fr (x=0)
        except: # Handle cases with too few points etc.
            B_guess = 0.0
            theta_guess = np.mean(edge_phase)


        # Override with user guesses if provided
        if init_guess is not None:
            fr_guess = init_guess.get('fr', fr_guess)
            Qi_guess = init_guess.get('Qi', Qi_guess)
            Qc_mag_guess = init_guess.get('Qc_mag', Qc_mag_guess)
            phi_guess = init_guess.get('phi', phi_guess)
            baseline_A_guess = init_guess.get('A', baseline_A_guess)
            theta_guess = init_guess.get('theta', theta_guess)
            B_guess = init_guess.get('B', B_guess)
        # --- End Initial Guesses ---

        # Define parameters with better bounds
        params.add('fr', value=fr_guess, min=frequency[0]*0.99, max=frequency[-1]*1.01) # Allow slightly outside range
        params.add('Qi', value=Qi_guess, min=1.0)
        params.add('Qc_mag', value=Qc_mag_guess, min=1.0)
        params.add('phi', value=phi_guess, min=-np.pi/2, max=np.pi/2) # Tighter, physical bounds
        # Add 'A', 'theta', 'B' based on fit_background flag
        if fit_background:
            params.add('A', value=baseline_A_guess, min=0)
            params.add('theta', value=theta_guess)
            params.add('B', value=B_guess) # B can be positive or negative
        else:
            params.add('A', value=baseline_A_guess, vary=False)
            params.add('theta', value=theta_guess, vary=False)
            params.add('B', value=B_guess, vary=False)

        # Perform the fit
        out = minimize(
            residual_circlefit_qi_qc,
            params,
            args=(frequency, data_real, data_imag),
            method='least_squares',
            nan_policy='omit' # Handle potential NaNs from model gracefully
        )

        # Generate best fit model trace AFTER optimization
        best_fit_model = np.full_like(frequency, np.nan, dtype=complex)
        if out.success:
            try:
                best_fit_model = complex_s21_model_qi_qc(frequency, **out.params.valuesdict())
            except Exception as e:
                print(f"Warning: Could not evaluate best fit model after optimization: {e}")


        return out, best_fit_model

    initial_guess = {
        'fr': resonator_data.estimate['f0'],
        'Qi': resonator_data.estimate['Qi'],
        'Qc_mag': resonator_data.estimate['Qc'],
        'phi': resonator_data.estimate['phi'],
        'A': 1.0,
        'theta': 0.0,
        'B': 0.0
    }

    result, model = fit_resonator_circle_qi_qc(resonator_data.freq, resonator_data.s21,
                                                init_guess=initial_guess,
                                            fit_background=True)
    Qi = result.params['Qi'].value
    Qc = result.params['Qc_mag'].value
    fo = result.params['fr'].value
    phi = result.params['phi'].value
    theta = result.params['theta'].value
    A = result.params['A'].value
    B = result.params['B'].value
    # Calculate total quality factor from internal and coupling Q
    # Qt = (Qi*Qc)/(Qi+Qc) - standard formula from parallel circuit analogy
    Qt = (Qi * Qc) / (Qi + Qc)
    
    # Calculate model predictions for both inverse S21 and S21
    model_is21 = 1/model
    model_s21 = model  # Convert inverse S21 to S21

    # (Optionally) replicate final circle shift from MATLAB, if you do that
    # deltax = (resonator_data.freq - fo)/fo
    # model_is21 = model_is21 * np.exp(-1j*((-theta) + (B*1e3)*deltax))
    # model_s21  = 1.0 / model_is21
    
    # Calculate goodness of fit metrics for reporting
    # Magnitude fits
    S21_mag = np.abs(resonator_data.s21)
    S21_phase = np.unwrap(np.angle(resonator_data.s21))
    iS21_mag = np.abs(resonator_data.is21)
    iS21_phase = np.unwrap(np.angle(resonator_data.is21))
    
    model_S21_mag = np.abs(model_s21)
    model_S21_phase = np.unwrap(np.angle(model_s21))
    model_iS21_mag = np.abs(model_is21)
    model_iS21_phase = np.unwrap(np.angle(model_is21))
    
    # Calculate root mean square error (RMSE) for magnitude and phase
    s21mag_fit_rmse = np.sqrt(np.mean((S21_mag - model_S21_mag)**2))
    s21phase_fit_rmse = np.sqrt(np.mean((S21_phase - model_S21_phase)**2))
    is21mag_fit_rmse = np.sqrt(np.mean((iS21_mag - model_iS21_mag)**2))
    is21phase_fit_rmse = np.sqrt(np.mean((iS21_phase - model_iS21_phase)**2))
    
    # Calculate circle fit RMSE (Euclidean distance in complex plane)
    xy_s21 = np.column_stack((np.real(resonator_data.s21), np.imag(resonator_data.s21)))
    xy_is21 = np.column_stack((np.real(resonator_data.is21), np.imag(resonator_data.is21)))
    xy_model_s21 = np.column_stack((np.real(model_s21), np.imag(model_s21)))
    xy_model_is21 = np.column_stack((np.real(model_is21), np.imag(model_is21)))
    
    s21circle_fit_rmse = np.sqrt(np.mean(np.sum((xy_s21 - xy_model_s21)**2, axis=1)))
    is21circle_fit_rmse = np.sqrt(np.mean(np.sum((xy_is21 - xy_model_is21)**2, axis=1)))
    
    # Calculate confidence intervals from covariance matrix
    # Get the covariance matrix - approach based on standard statistical methods
    N = len(resonator_data.freq) * 2  # Number of data points (real + imag)
    p = len(result.params)  # Number of parameters
    
    # Calculate parameter standard errors from covariance matrix
    # This uses standard statistical formulas for non-linear least squares
    pcov = result.covar  # Get covariance matrix directly from result
    perr = np.sqrt(np.diag(pcov))  # Standard errors are sqrt of diagonal
    
    # Scale by 1.96 for ~95% confidence interval (assumes normal distribution)
    QiError = perr[0] * 1.96  # Scale back up for internal Q
    QcError = perr[1] * 1.96  # Scale back up for coupling Q
    foError = perr[2] * 1.96   # Scale back up for frequency
    phiError = perr[3] * 1.96       # Phase error
    thetaError = perr[4] * 1.96     # Rotation angle error
    AError = perr[5] * 1.96         # Amplitude scaling error
    BError = perr[6] * 1.96  # Background error (scaled)
    
    # Calculate QtError using error propagation formula
    # For Qt = (Qi*Qc)/(Qi+Qc), apply standard error propagation
    QtError = Qt * np.sqrt((QiError/Qi)**2 + (QcError/Qc)**2)
        
    
    # Store fit results in the resonator_data object
    fit_results = {
        # Main resonator parameters
        "Qi": Qi,              # Internal quality factor
        "Qc": Qc,              # Coupling quality factor
        "Qt": Qt,              # Total quality factor
        "fo": fo,              # Resonance frequency (Hz)
        
        # Additional model parameters
        "phi": phi,            # Coupling phase (radians)
        "theta": theta,        # Global rotation angle (radians)
        "A": A,                # Amplitude scaling factor
        "B": B,                # Linear background term
        
        # Parameter uncertainties (95% confidence intervals)
        "QiError": QiError,    # Internal Q error
        "QcError": QcError,    # Coupling Q error
        "QtError": QtError,    # Total Q error
        "foError": foError,    # Frequency error (Hz)
        "phiError": phiError,  # Coupling phase error
        "thetaError": thetaError,  # Rotation angle error
        "AError": AError,      # Amplitude scaling error
        "BError": BError,      # Background term error
        
        # Model predictions for plotting and analysis
        "model_s21": model_s21,     # Model prediction for S21
        "model_is21": model_is21,   # Model prediction for inverse S21
        
        # Goodness of fit metrics
        "gof": {
            "S21magFitRMSE": s21mag_fit_rmse,      # Magnitude RMSE for S21
            "S21PhaseFitRMSE": s21phase_fit_rmse,  # Phase RMSE for S21
            "iS21magFitRMSE": is21mag_fit_rmse,    # Magnitude RMSE for inverse S21
            "iS21PhaseFitRMSE": is21phase_fit_rmse,  # Phase RMSE for inverse S21
            "S21CircleFitRMSE": s21circle_fit_rmse,  # Complex plane RMSE for S21
            "iS21CircleFitRMSE": is21circle_fit_rmse,  # Complex plane RMSE for inverse S21
        }
    }

    if not hasattr(resonator_data, 'fit'):
        resonator_data.fit = {}
    
    # Update with new results
    resonator_data.fit.update(fit_results)


    return resonator_data


def _consolidate_results(resonator_data: ResonatorData) -> None:
    """
    Create a formatted results summary from fit parameters and metadata.
    
    This function generates a human-readable summary of the fitting results,
    formatting quality factors, frequencies, and other parameters with 
    appropriate units and significant figures. The summary follows the exact
    format used in the MATLAB version for consistency.
    
    Args:
        resonator_data: ResonatorData object with fit results and metadata
        
    Returns:
        None - Results are stored in resonator_data.fit["spectral_fit_results_summary"]
    
    Note:
        The function handles different magnitude ranges for quality factors,
        adding appropriate unit suffixes (k for thousands, M for millions).
    """
    # Extract quality factors and their errors for formatting
    qt = resonator_data.fit["Qt"]
    qt_err = resonator_data.fit["QtError"]
    qi = resonator_data.fit["Qi"]
    qi_err = resonator_data.fit["QiError"]
    qc = resonator_data.fit["Qc"]
    qc_err = resonator_data.fit["QcError"]
    
    # Initialize summary list
    summary = []
    
    # Add filename if available
    if hasattr(resonator_data, "filename"):
        summary.append(f"{resonator_data.filename}")
    
    summary.append("Spectral Fit Results")
    
    # Add temperature if available
    if hasattr(resonator_data, "temp"):
        summary.append(f"  Temp = {resonator_data.temp:.3f} mK")
    
    # Add attenuation - fixed to use %.0f format exactly like MATLAB
    summary.append(f"  Applied Attenuation = {resonator_data.atten:.0f} dB")
    
    # Add photon number - important for power dependence studies
    summary.append(f"  <n_p> = {resonator_data.Np:.2e}")
    
    # Add resonance frequency - MATLAB uses 9 decimal places
    summary.append(f"  fo = {resonator_data.fit['fo']/1e9:.9f} GHz")
    
    # Format Q values based on magnitude - exactly matching MATLAB's logic
    # This makes the output more readable by using appropriate units
    if qi > 1e5:
        # For very high Q resonators, use megaunits (M)
        summary.append(f"  Q_t = {qt/1e6:.3f} +/- {qt_err/1e6:.2g} M")
        summary.append(f"  Q_i = {qi/1e6:.3f} +/- {qi_err/1e6:.2g} M")
        summary.append(f"  Q_c = {qc/1e6:.3f} +/- {qc_err/1e6:.2g} M")
    elif qi > 1e3:
        # For mid-range Q resonators, use kilounits (k)
        summary.append(f"  Q_t = {qt/1e3:.3f} +/- {qt_err/1e3:.2g} k")
        summary.append(f"  Q_i = {qi/1e3:.3f} +/- {qi_err/1e3:.2g} k")
        summary.append(f"  Q_c = {qc/1e3:.3f} +/- {qc_err/1e3:.2g} k")
    else:
        # For low Q resonators, use plain numbers
        summary.append(f"  Q_t = {qt:.3f} +/- {qt_err:.2g}")
        summary.append(f"  Q_i = {qi:.3f} +/- {qi_err:.2g}")
        summary.append(f"  Q_c = {qc:.3f} +/- {qc_err:.2g}")
    
    # Add other fit parameters - match MATLAB tab spacing
    # These parameters determine the resonator circle shape and position
    summary.append(
        f"  φ = {resonator_data.fit['phi']:.3f} +/- {resonator_data.fit['phiError']:.3f}    "
        f"  θ = {resonator_data.fit['theta']:.3f}+/- {resonator_data.fit['thetaError']:.3f}"
    )
    summary.append(
        f"  A = {resonator_data.fit['A']:.3f}+/- {resonator_data.fit['AError']:.3g}         "
        f"B = {resonator_data.fit['B']:.3g}+/- {resonator_data.fit['BError']:.3g}"
    )
    
    # Handle SNR metrics - MATLAB uses specific order
    # SNR is important for evaluating measurement quality
    s21mag_snr = resonator_data.s21mag_snr
    is21mag_snr = resonator_data.is21mag_snr
    
    # For circular SNR, extract the first value if it's a tuple
    if isinstance(resonator_data.s21circ_snr, tuple):
        s21circ_snr = resonator_data.s21circ_snr[0]
    else:
        s21circ_snr = resonator_data.s21circ_snr
        
    if isinstance(resonator_data.is21circ_snr, tuple):
        is21circ_snr = resonator_data.is21circ_snr[0]
    else:
        is21circ_snr = resonator_data.is21circ_snr
    
    # Add SNR metrics exactly as in MATLAB
    summary.append(
        f"magSNR:   S21^-1 = {s21mag_snr:.3f}       "
        f"S21 = {is21mag_snr:.3f} "
    )
    summary.append(
        f"circSNR:   S21^-1 = {is21circ_snr:.3f}       "
        f"S21 = {s21circ_snr:.3f}"
    )
    
    # Add goodness of fit metrics exactly as in MATLAB
    # Lower RMSE values indicate better fits
    gof = resonator_data.fit["gof"]
    summary.append(
        f"  Mag Fit RMSE   S21^-1: {gof['iS21magFitRMSE']:.3f}     "
        f"S21: {gof['S21magFitRMSE']:.3f}"
    )
    summary.append(
        f"  Phase Fit RMSE S21^-1: {gof['iS21PhaseFitRMSE']:.3f}     "
        f"S21: {gof['S21PhaseFitRMSE']:.3f} "
    )
    summary.append(
        f"  Circle Fit RMSE S21^-1: {gof['iS21CircleFitRMSE']:.3f}    "
        f"S21: {gof['S21CircleFitRMSE']:.3f} "
    )
    
    # Add measurement metadata if available - exactly as in MATLAB
    if hasattr(resonator_data, "measured_date"):
        summary.append(f"\nMeasured Date= {resonator_data.measured_date}")
    
    if hasattr(resonator_data, "avenum"):
        summary.append(f"Num of sweep = {resonator_data.avenum}")
    
    # Add power information - important for power dependence studies
    summary.append(
        f" launchPower = {resonator_data.launch_power:4d} dBm       "
        f"systemAttenuation = {resonator_data.system_attenuation:4.1f} dB"
    )
    
    # Add raw S21 magnitude range if available - exactly as in MATLAB
    if hasattr(resonator_data, "max_raw_s21") and hasattr(resonator_data, "min_raw_s21"):
        summary.append(
            f"max(|rawS21|={resonator_data.max_raw_s21:4.1f} dBm       "
            f"min(|rawS21|={resonator_data.min_raw_s21:4.1f} dBm"
        )
    
    # Add fit mode tag if available
    if hasattr(resonator_data, "fit"):
        fit_mode_tag = f"FitMode = {resonator_data.fit['model']}"
        summary.append(fit_mode_tag)
    
    # Add extra comment if available
    if hasattr(resonator_data, "extra_comment"):
        summary.append(resonator_data.extra_comment)
    
    # Store the summary in the fit dictionary
    resonator_data.fit["spectral_fit_results_summary"] = summary


def resonator_fit_plot(resonator_data: ResonatorData) -> None:
    """
    Generate a comprehensive visualization of resonator fit results.
    
    This function creates a figure with six subplots for resonator data visualization,
    and includes a text area for displaying fit results summary:
    1. S21 magnitude vs frequency (top left)
    2. S21 phase vs frequency (top right)
    3. S21 parametric circle plot in complex plane (middle left)
    4. Inverse S21 parametric circle plot in complex plane (middle right)
    5. Inverse S21 magnitude vs frequency (bottom left)
    6. Inverse S21 phase vs frequency (bottom right)
    
    Both measured data and model fits are shown for comparison, and the fit summary
    text is displayed in the figure for easy reference.
    
    Args:
        resonator_data: ResonatorData object with measurement data and fit results
    
    Returns:
        None - Displays and optionally saves the plot
    """
    # Create a figure with GridSpec for flexible subplot layout
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3)  # 3x3 grid for plots and text
    
    # Convert frequency to GHz for readability
    freq_ghz = resonator_data.freq / 1e9
    
    # Plot 1: |S21| vs frequency - shows magnitude response of resonator
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot measured data points
    ax1.plot(freq_ghz, 20*np.log10(np.abs(resonator_data.s21)), 'bo', 
             markersize=2, label='Measured')
    
    # Plot model fit line
    if "model_s21" in resonator_data.fit:
        ax1.plot(freq_ghz, 20*np.log10(np.abs(resonator_data.fit["model_s21"])), 
                 'r-', label='Fit')
    
    # Set axis labels and style
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('|S21| (dB)')
    ax1.set_title('S21 Magnitude')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Phase vs frequency - shows phase response of resonator
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot measured phase data (unwrapped to avoid 2π jumps)
    ax2.plot(freq_ghz, np.unwrap(np.angle(resonator_data.s21)), 'bo', 
             markersize=2, label='Measured')
    
    # Plot model phase fit
    if "model_s21" in resonator_data.fit:
        ax2.plot(freq_ghz, np.unwrap(np.angle(resonator_data.fit["model_s21"])), 
                 'r-', label='Fit')
    
    # Set axis labels and style
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (rad)')
    ax2.set_title('S21 Phase')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Real vs Imaginary (parametric circle) - key for resonator analysis
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Plot measured S21 points in complex plane
    ax3.plot(np.real(resonator_data.s21), np.imag(resonator_data.s21), 
             'bo', markersize=2, label='Measured')
    
    # Plot model S21 circle
    if "model_s21" in resonator_data.fit:
        ax3.plot(np.real(resonator_data.fit["model_s21"]), 
                 np.imag(resonator_data.fit["model_s21"]), 
                 'r-', label='Fit')
    
    # Set axis labels and style
    ax3.set_xlabel('Re(S21)')
    ax3.set_ylabel('Im(S21)')
    ax3.set_title('S21 Parametric Circle')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')  # Equal aspect ratio ensures circle appears as circle
    
    # Plot 4: Inverse S21 parametric circle plot in complex plane
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot measured inverse S21 points in complex plane
    ax4.plot(np.real(resonator_data.is21), np.imag(resonator_data.is21), 
             'bo', markersize=2, label='Measured')
    
    # Plot model inverse S21 circle
    if "model_is21" in resonator_data.fit:
        ax4.plot(np.real(resonator_data.fit["model_is21"]), 
                 np.imag(resonator_data.fit["model_is21"]), 
                 'r-', label='Fit')
    
    # Set axis labels and style
    ax4.set_xlabel('Re(S21^{-1})')
    ax4.set_ylabel('Im(S21^{-1})')
    ax4.set_title('Inverse S21 Parametric Circle')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')  # Equal aspect ratio ensures circle appears as circle
    
    # Plot 5: Inverse S21 magnitude vs frequency
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Plot measured inverse S21 magnitude
    ax5.plot(freq_ghz, 20*np.log10(np.abs(resonator_data.is21)), 'bo', 
             markersize=2, label='Measured')
    
    # Plot model inverse S21 magnitude
    if "model_is21" in resonator_data.fit:
        ax5.plot(freq_ghz, 20*np.log10(np.abs(resonator_data.fit["model_is21"])), 
                 'r-', label='Fit')
    
    # Set axis labels and style
    ax5.set_xlabel('Frequency (GHz)')
    ax5.set_ylabel('|S21^{-1}| (dB)')
    ax5.set_title('Inverse S21 Magnitude')
    ax5.legend()
    ax5.grid(True)
    
    # Plot 6: Inverse S21 phase vs frequency
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Plot measured inverse S21 phase
    ax6.plot(freq_ghz, np.unwrap(np.angle(resonator_data.is21)), 'bo', 
             markersize=2, label='Measured')
    
    # Plot model inverse S21 phase
    if "model_is21" in resonator_data.fit:
        ax6.plot(freq_ghz, np.unwrap(np.angle(resonator_data.fit["model_is21"])), 
                 'r-', label='Fit')
    
    # Set axis labels and style
    ax6.set_xlabel('Frequency (GHz)')
    ax6.set_ylabel('Phase (rad)')
    ax6.set_title('Inverse S21 Phase')
    ax6.legend()
    ax6.grid(True)
    
    # Add text summary in a subplot that spans the right column
    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.axis('off')  # Turn off axes
    
    # Add fit summary text
    if "spectral_fit_results_summary" in resonator_data.fit:
        summary_text = '\n'.join(resonator_data.fit["spectral_fit_results_summary"])
        ax_text.text(0, 0.95, summary_text, verticalalignment='top', 
                     fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save the plot if requested
    if resonator_data.save_to_info["save_plots"]:
        save_plot(fig, resonator_data)
        print(f'Plot saved to {resonator_data.save_to_info["file_path"]}')
    
    if resonator_data.save_to_info["show_plots"]:
        # Show the plot on screen
        plt.show()
    else:
        # Close the plot to avoid displaying it
        plt.close(fig)


def save_plot(fig: plt, 
              resonator_data: ResonatorData, 
              attributes: dict = None) -> None:
    """
    Saves a matplotlib figure to an HDF5 file with optional metadata attributes.
    
    This function converts a matplotlib figure to a PNG image, stores it in an HDF5 file,
    and optionally associates metadata attributes with the image dataset. The function
    organizes plots in groups within the HDF5 file for better organization.
    
    Args:
        fig: A matplotlib Figure object containing the plot to save
        resonator_data: A ResonatorData object containing save_to_info dictionary with:
            - 'plot_group': The HDF5 group path where the plot should be saved
            - 'file_path': The path to the HDF5 file
            - 'plot_name_format': Either 'auto' for timestamp-based naming or another value for custom naming
            - 'plot_name': Custom name for the plot (used if plot_name_format is not 'auto')
        attributes: Optional dictionary of metadata to store with the plot.
                   Can include experimental parameters, notes, or any other relevant information.
    
    Returns:
        None
    
    Notes:
        - Creates the HDF5 file if it doesn't exist
        - Creates the group if it doesn't exist
        - Converts the figure to PNG format with 300 DPI resolution
        - The plot is stored as a dataset of uint8 values representing the binary image data
        - If the HDF5 file is not accessible (e.g., permissions issues, corrupted file),
          an exception will be raised by h5py
    
    Example usage:
        save_plot(fig, resonator_data, {'temperature': 20, 'power': -30, 'notes': 'Good measurement'})
    """
    group = resonator_data.save_to_info['plot_group']
    save_path = Path(resonator_data.save_to_info['file_path'])
    
    if resonator_data.save_to_info["plot_name_format"] == 'auto':
        plot_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        plot_name = resonator_data.save_to_info["plot_name"]

    if resonator_data.save_to_info["file_type"] == 'hdf5':
        buf = io.BytesIO() # Create a buffer to save the figure
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight') # Save to buffer
        buf.seek(0) # Rewind the buffer to the beginning

        plot_data = buf.getvalue() # Get the byte data from the buffer


        with h5py.File(save_path, 'a') as f:
            if group not in f:
                plots_group = f.create_group(group)
            else:
                plots_group = f[group]
            
            plot_dataset = plots_group.create_dataset(plot_name, data=np.frombuffer(plot_data, dtype=np.uint8))
            # Add metadata
            if attributes is not None:
                for key, value in attributes.items():
                    plot_dataset.attrs[key] = value
    
    elif resonator_data.save_to_info["file_type"] == 'png':
        fig.savefig(save_path / (plot_name + '.png' ), format='png', dpi=300, bbox_inches='tight')
        

def load_plot(file_path, group, name=None):
    """
    Load and display plots stored in an HDF5 file with their metadata.
    
    This function can either display a specific plot by name or all plots
    in a given group, along with their associated metadata attributes.
    
    Args:
        file_path: Path to the HDF5 file containing the plots
        group: Group path within the HDF5 file where plots are stored
        name: Optional name of a specific plot to display. If None, displays all plots.
    
    Returns:
        None - Displays the plots and prints their metadata
        
    Notes:
        - Each plot is displayed in a separate figure
        - Metadata attributes are printed to the console
        - If group or plot name doesn't exist, appropriate error messages are printed
    
    Example usage:
        # Display a specific plot
        load_plot('results.h5', 'daily_measurements', '20220315_120000')
        
        # Display all plots in a group
        load_plot('results.h5', 'daily_measurements')
    """
    
    with h5py.File(file_path, 'r') as f:
        if group not in f:
            print(f"No {group} found in the file.")
            return
        
        plots_group = f[group]
        
        if name is not None:
            # Display a specific plot
            if name in plots_group:
                plot_data = plots_group[name][...]
                img = Image.open(io.BytesIO(plot_data.tobytes()))
                plt.figure(figsize=(10, 8))
                plt.imshow(np.array(img))
                plt.axis('off')
                plt.title(f"Plot: {name}")
                plt.show()
                
                # Display metadata
                print("Plot metadata:")
                for key, value in plots_group[name].attrs.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Plot {name} not found.")
        else:
            # Display all plots
            plot_ids = list(plots_group.keys())
            for i, pid in enumerate(plot_ids):
                plot_data = plots_group[pid][...]
                img = Image.open(io.BytesIO(plot_data.tobytes()))
                
                plt.figure(figsize=(10, 8))
                plt.imshow(np.array(img))
                plt.axis('off')
                plt.title(f"Plot: {pid}")
                plt.show()
                
                # Display metadata
                print(f"Plot {pid} metadata:")
                for key, value in plots_group[pid].attrs.items():
                    print(f"  {key}: {value}")