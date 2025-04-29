import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from typing import List, Dict, Any, Optional
from collections import defaultdict
import datetime
import os
from pathlib import Path
import warnings

from resonance_py.data_analysis.resonator_data import ResonatorData



# --- Power-Dependent Qi Model ---

def power_qi_model(n: np.ndarray, QA: float, QTLS: float, nc: float, alpha: float) -> np.ndarray:
    """
    Calculates Qi based on the standard TLS loss model.

    Model: 1/Qi(n) = 1/QA + (1/QTLS) / (1 + (n / nc)**alpha)

    Args:
        n: Array of average photon numbers.
        QA: Quality factor due to power-independent technical loss.
        QTLS: Quality factor due to TLS loss at zero power.
        nc: Critical photon number characterizing TLS saturation.
        alpha: Exponent describing the power dependence of TLS loss.

    Returns:
        Array of calculated Qi values corresponding to input n.
    """
    # Ensure n is an array (handles scalar inputs too)
    n_array = np.asarray(n)

    # Calculate inverse Qi based on the model
    inv_qi = (1.0 / QA) + (1.0 / QTLS) / (1.0 + (n_array / nc)**alpha)

    if np.isscalar(inv_qi):
        # Scalar case - use simple conditional
        return np.inf if inv_qi <= 0 else 1.0 / inv_qi
    else:
        # Array case - use array indexing
        result = 1.0 / inv_qi  # Start with standard calculation
        result[inv_qi <= 0] = np.inf  # Replace invalid values
        return result

def inv_power_qi_model(n: np.ndarray, inv_QA: float, inv_QTLS: float, nc: float, alpha: float) -> np.ndarray:
    """
    Calculates inverse Qi (1/Qi) based on the standard TLS loss model.
    This form is often better for fitting.

    Model: 1/Qi(n) = inv_QA + inv_QTLS / (1 + (n / nc)**alpha)

    Args:
        n: Array of average photon numbers.
        inv_QA: Power-independent technical loss (1/QA).
        inv_QTLS: TLS loss at zero power (1/QTLS).
        nc: Critical photon number characterizing TLS saturation.
        alpha: Exponent describing the power dependence of TLS loss.

    Returns:
        Array of calculated 1/Qi values corresponding to input n.
    """
    return inv_QA + inv_QTLS / (1.0 + (np.asarray(n) / nc)**alpha)


# --- Power Fit Function (Equivalent to PdFit.m) ---

def pd_fit(
    n_photons: np.ndarray,
    qi_values: np.ndarray,
    qi_errors: Optional[np.ndarray] = None,
    photon_values_for_qi1: List[float] = [1.0],
    skip_indices: Optional[np.ndarray] = None,
    initial_guesses: Optional[Dict[str, float]] = None,
    param_bounds: Optional[Dict[str, tuple]] = None,
    smooth_data: bool = True,  # Added parameter for data smoothing
    robust_fit: bool = True,    # Added parameter for robust fitting
    power_weight_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Fits the power dependence of internal quality factor (Qi) using the TLS model.

    Args:
        n_photons: Array of average photon numbers.
        qi_values: Array of corresponding internal quality factors.
        qi_errors: Optional array of errors for Qi values (used for weighting fit).
        photon_values_for_qi1: Photon numbers at which to calculate Qi (e.g., [1.0]).
        skip_indices: Indices of data points to exclude from the fit.
        initial_guesses: Optional dictionary of initial guesses for parameters
                         (inv_QA, inv_QTLS, nc, alpha).
        param_bounds: Optional dictionary of bounds for parameters.
        smooth_data: Whether to apply smoothing to inverse Qi data before fitting.
        robust_fit: Whether to use robust fitting method (equivalent to LAR in MATLAB).

    Returns:
        Dictionary containing fit results:
        - fit_result: lmfit ModelResult object (or None if fit failed).
        - params: Dictionary of fitted parameters (QA, QTLS, nc, alpha).
        - errors: Dictionary of parameter standard errors.
        - qi_at_photons: Dictionary mapping photon value to calculated Qi.
        - selected_indices: Indices of data points used in the fit.
        - skipped_indices: Indices of data points skipped.
        - model_qi: Qi values predicted by the model for the *original* input n_photons.
        - success: Boolean indicating if the fit was successful.
        - error_message: String containing error message if fit failed.
    """
    results = {"success": False, "error_message": "Initialization failed"}
    selected_indices = np.arange(len(n_photons))

    # --- Data Selection ---
    # Remove NaN/inf values initially
    valid_mask = ~np.isnan(n_photons) & ~np.isnan(qi_values) & np.isfinite(n_photons) & np.isfinite(qi_values) & (qi_values > 0)
    selected_indices = selected_indices[valid_mask]

    # Apply user-provided skip indices
    if skip_indices is not None:
        selected_indices = np.setdiff1d(selected_indices, skip_indices, assume_unique=True)

    # Check if enough points remain for fitting
    if len(selected_indices) < 4: # Need at least 4 points for 4 parameters
        results["error_message"] = f"Not enough valid data points ({len(selected_indices)}) for fitting."
        results["selected_indices"] = selected_indices
        results["skipped_indices"] = np.setdiff1d(np.arange(len(n_photons)), selected_indices)
        return results

    n_fit = n_photons[selected_indices]
    qi_fit = qi_values[selected_indices]
    inv_qi_fit = 1.0 / qi_fit

    # --- Apply smoothing to inverse Qi values (similar to MATLAB smooth function) ---
    if smooth_data and len(inv_qi_fit) > 2:
        # Simple moving average with window size 2 (like MATLAB smooth(y, 2))
        from scipy.signal import savgol_filter
        try:
            # Use Savitzky-Golay filter for smoothing (more advanced than moving average)
            # window_length must be odd, so use 3 for window size ~2
            inv_qi_fit = savgol_filter(inv_qi_fit, min(3, len(inv_qi_fit)), 1)
        except Exception as e:
            print(f"Warning: Smoothing failed, using original data - {e}")

    # --- Weights for Fitting ---
    # weights = None
    # if qi_errors is not None:
    #     qi_err_fit = qi_errors[selected_indices]
    #     valid_err_mask = ~np.isnan(qi_err_fit) & np.isfinite(qi_err_fit) & (qi_err_fit > 0)
    #     if np.any(valid_err_mask):
    #         # Propagate error: err(1/Qi) = err(Qi) / Qi^2
    #         inv_qi_err_fit = np.full_like(qi_err_fit, np.nan)
    #         inv_qi_err_fit[valid_err_mask] = qi_err_fit[valid_err_mask] / (qi_fit[valid_err_mask]**2)
    #         # Use inverse variance as weights, ignore points with invalid errors
    #         weights = np.zeros_like(inv_qi_err_fit)
    #         valid_weight_mask = ~np.isnan(inv_qi_err_fit) & (inv_qi_err_fit > 0)
    #         weights[valid_weight_mask] = 1.0 / (inv_qi_err_fit[valid_weight_mask]**2)
    #         # Ensure weights are finite
    #         weights[~np.isfinite(weights)] = 0
    #         if np.sum(weights > 0) < 4: # Check if enough points have valid weights
    #              warnings.warn("Not enough points with valid errors for weighted fit. Performing unweighted fit.")
    #              weights = None # Revert to unweighted fit

    
    # # Add power weighting to emphasize low photon numbers
    # power_weighting = True  # Add as function parameter or hardcode
    # power_weight_factor = 0  # Adjust to control emphasis (higher = more emphasis)

    # if power_weighting:
    #     # Create weights inversely proportional to photon number
    #     power_weights = (np.min(n_fit) / n_fit)**power_weight_factor
        
    #     # Combine with existing error-based weights if any
    #     if weights is not None:
    #         # Normalize both weight sets to avoid scaling issues
    #         norm_weights = weights / np.max(weights) if np.max(weights) > 0 else weights
    #         norm_power_weights = power_weights / np.max(power_weights)
    #         # Multiply weights (both contribute to final weighting)
    #         weights = norm_weights * norm_power_weights
    #     else:
    #         weights = power_weights
        
    #     print(f"  Applied power weighting (factor={power_weight_factor}) to emphasize low-power fit")
    # 1. start with 'None'  ->  ordinary least squares
# -------------------------------------------------------------------
#  Build the weight vector 'weights'
# -------------------------------------------------------------------
# Start with None  ->  ordinary least squares
    weights = None               # (will become a NumPy array later)

    # -------------------------------------------------
    # (A)  Error-based weights  (1/σ² for 1/Qi if errors given)
    # -------------------------------------------------
    if qi_errors is not None:
        qi_err_fit = qi_errors[selected_indices]

        # propagate:  σ(1/Q) = σ(Q) / Q²
        inv_qi_err = np.where(
            (qi_err_fit > 0) & np.isfinite(qi_err_fit),
            qi_err_fit / (qi_fit**2),
            np.nan
        )

        # keep the points that *do* have a finite error
        if np.isfinite(inv_qi_err).sum() >= 4:
            w_err            = 1.0 / inv_qi_err**2
            w_err[~np.isfinite(w_err)] = 0.0          # guard nan / inf
            w_err /= np.sqrt(np.mean(w_err**2))       # RMS-normalise
            weights = w_err                            # first weight set

    gamma = power_weight_factor          # <----- choose your factor here  (float)

    # -------------------------------------------------
    # (B)  Decade-balancing weights  √(n / n_min)
    #     makes every log-decade speak with similar volume
    # -------------------------------------------------


    # -------------------------------------------------
    # (C)  Extra low-power emphasis -------------------
    #     γ = 0  → none
    #     γ = 1  → inverse-n
    #     γ = 2  → inverse-n² (very aggressive)
    # -------------------------------------------------
    

    if gamma != 0:
        w_pow = (n_fit.min() / n_fit)**gamma
        w_pow /= np.sqrt(np.mean(w_pow**2))           # RMS-normalise
        w_dec = np.sqrt(n_fit / n_fit.min())
        w_dec /= np.sqrt(np.mean(w_dec**2))               # RMS-normalise
    else:
        w_pow = 1.0                                   # broadcast scalar
        w_dec = 1.0

    # -------------------------------------------------
    # Combine the three weight sets
    # -------------------------------------------------
    if weights is None:
        weights = np.ones_like(n_fit, dtype=float)    # start from ones

    weights *= w_dec * w_pow                          # element-wise product
    # weights /= np.sqrt(np.mean(weights**2))

    # Fall back to unweighted if something went wrong
    if not np.any(weights > 0):
        weights = None

    # -------------------------------------------------------------------
    
    # Check number of valid points and use appropriate method (like in PdFit.m)
    if len(selected_indices) > 3:
        # Continue with normal TLS model fit (existing code)
        # --- Fitting Setup ---
        pmodel = Model(inv_power_qi_model)
        params = Parameters()

        # Define default initial guesses and bounds based on data like in PdFit_core.m
        # Calculate data-driven bounds similar to MATLAB implementation
        max_qi = np.max(qi_fit)
        min_qi = np.min(qi_fit)
        median_inv_qi = np.median(inv_qi_fit)
        median_n = np.median(n_fit)
        max_n = np.max(n_fit)

        low_q  = np.percentile(qi_fit, 25)          # lower quartile of Qi
        high_q = np.percentile(qi_fit, 75)          # upper quartile of Qi
        inv_QTLS_guess = 1.0 / max(low_q,  1e3)
        inv_QA_guess   = 1.0 / max(high_q, 1e3)
        nc_guess       = np.exp(np.mean(np.log(n_fit)))   # geometric mean
        alpha_guess    = 0.5
        


        # MATLAB-equivalent initial guesses (after inverting)
        default_guesses = {
            'inv_QA':   inv_QA_guess,
            'inv_QTLS': inv_QTLS_guess,
            'nc':       nc_guess,
            'alpha':    alpha_guess
            }

        # MATLAB-equivalent bounds (already correct in your implementation)
        data_bounds = {
            'inv_QA': (1.0/(100*max_qi), 1.0/max_qi),
            'inv_QTLS': (1.0/(10*min_qi), 1.0/min_qi), 
            'nc': (1e-3, max_n),
            'alpha': (0.5, 1.0)
        }
        
        # Use more conservative bounds if not overridden
        default_bounds = {
            'inv_QA': (1e-9, 1e-3),
            'inv_QTLS': (1e-9, 1e-3),
            'nc': (1e-3, 1e12),
            'alpha': (0.5, 1.0)
        }

        # Apply user overrides for guesses and bounds
        current_guesses = initial_guesses if initial_guesses else default_guesses
        
        # Prioritize user bounds, then data-driven bounds, then default bounds
        if param_bounds is not None:
            current_bounds = param_bounds.copy()
            # Fill in missing bounds with data-driven values
            for param in default_bounds:
                if param not in current_bounds:
                    current_bounds[param] = data_bounds.get(param, default_bounds[param])
        else:
            current_bounds = data_bounds

        # Add parameters to lmfit object
        for name, guess in default_guesses.items():
            val = current_guesses.get(name, guess)
            b_min, b_max = current_bounds.get(name, default_bounds[name])
            params.add(name, value=val, min=b_min, max=b_max)

        # --- Perform Fit with MATLAB-like method options ---
        # Set method and options similar to MATLAB's Trust-Region algorithm
        fit_kws = {
            'ftol': 1e-3,              # Exactly match MATLAB's TolFun
            'xtol': 1e-3,              # Similar precision for parameter changes
            'gtol': 1e-3,              # Similar precision for gradient
        }

        # Add robust fitting if requested (similar to MATLAB's LAR)
        if robust_fit:
            fit_kws['loss'] = 'soft_l1'  # Most similar to MATLAB's LAR

        # Use try-except to handle potential fitting errors with these parameters
        try:
            # First attempt with exact MATLAB settings
            fit_result = pmodel.fit(inv_qi_fit, params, n=n_fit, weights=weights, 
                                  method='least_squares',  # Most similar to MATLAB's Trust-Region
                                  max_nfev=5000,
                                  fit_kws=fit_kws
                                  )
            if not fit_result.success:
                 raise RuntimeError(f"lmfit optimization failed: {fit_result.message}")

            # --- Extract Results ---
            fit_params = fit_result.params
            inv_QA_fit = fit_params['inv_QA'].value
            inv_QTLS_fit = fit_params['inv_QTLS'].value
            nc_fit = fit_params['nc'].value
            alpha_fit = fit_params['alpha'].value

            # Calculate QA and QTLS (handle potential division by zero)
            QA = 1.0 / inv_QA_fit if inv_QA_fit != 0 else np.inf
            QTLS = 1.0 / inv_QTLS_fit if inv_QTLS_fit != 0 else np.inf

            # Calculate Qi at specified photon values
            qi_at_photons = {}
            for n_val in photon_values_for_qi1:
                qi_at_photons[n_val] = power_qi_model(n_val, QA, QTLS, nc_fit, alpha_fit)

            # Calculate parameter errors (propagate for QA, QTLS)
            errors = {}
            for name in ['inv_QA', 'inv_QTLS', 'nc', 'alpha']:
                stderr = fit_params[name].stderr
                errors[name] = stderr if stderr is not None else np.nan # Store inv_QA/inv_QTLS errors first

            # Propagate errors carefully
            if not np.isnan(errors['inv_QA']) and inv_QA_fit != 0:
                errors['QA'] = abs(errors['inv_QA'] / (inv_QA_fit**2))
            else:
                errors['QA'] = np.nan

            if not np.isnan(errors['inv_QTLS']) and inv_QTLS_fit != 0:
                errors['QTLS'] = abs(errors['inv_QTLS'] / (inv_QTLS_fit**2))
            else:
                errors['QTLS'] = np.nan

            # Calculate model prediction for *all* original input Np values
            model_qi = power_qi_model(n_photons, QA, QTLS, nc_fit, alpha_fit)

            # Generate smooth fitted curve for plotting (similar to MATLAB)
            n_model = np.logspace(min(0.1, np.log10(np.min(n_fit))), np.log10(np.max(n_fit)), 50)
            qi_model = power_qi_model(n_model, QA, QTLS, nc_fit, alpha_fit)
            
            results = {
                "fit_result": fit_result,
                "params": {"QA": QA, "QTLS": QTLS, "nc": nc_fit, "alpha": alpha_fit},
                "errors": errors, # Contains errors for QA, QTLS, nc, alpha, inv_QA, inv_QTLS
                "qi_at_photons": qi_at_photons,
                "selected_indices": selected_indices,
                "skipped_indices": np.setdiff1d(np.arange(len(n_photons)), selected_indices),
                "model_qi": model_qi,
                "model_curve": {"n": n_model, "qi": qi_model},  # Added for plotting
                "success": True,
                "error_message": None
            }

        except Exception as e:
            # Store failure information
            results = {
                "fit_result": None,
                "params": None,
                "errors": None,
                "qi_at_photons": None,
                "selected_indices": selected_indices,
                "skipped_indices": np.setdiff1d(np.arange(len(n_photons)), selected_indices),
                "model_qi": np.full_like(n_photons, np.nan),
                "success": False,
                "error_message": f"Power fitting failed: {e}"
            }
            print(results["error_message"]) # Print error during execution

        return results
    elif len(selected_indices) in [2, 3]:
        # For 2-3 points, use simple linear fit instead (like MATLAB)
        try:
            n_fit = n_photons[selected_indices]
            qi_fit = qi_values[selected_indices]
            inv_qi_fit = 1.0 / qi_fit
            
            # Linear fit to 1/Qi vs n
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(n_fit, inv_qi_fit)
            
            # Calculate equivalent parameters
            QA = 1.0 / intercept if intercept > 0 else np.inf
            
            # For simple model, set defaults for other parameters
            QTLS = min(qi_fit) # Approximation
            nc = 1.0
            alpha = 0.5
            
            # Generate model curve
            n_model = np.logspace(np.log10(np.min(n_fit)), np.log10(np.max(n_fit)), 50)
            model_qi = 1.0 / (slope * n_model + intercept)
            
            # Calculate Qi at specified photon values
            qi_at_photons = {}
            for n_val in photon_values_for_qi1:
                qi_at_photons[n_val] = 1.0 / (slope * n_val + intercept)
            
            results = {
                "fit_result": None,  # No lmfit result for this simple fit
                "params": {"QA": QA, "QTLS": QTLS, "nc": nc, "alpha": alpha,
                          "p1": slope, "p2": intercept},  # Store linear fit params like MATLAB
                "errors": {},
                "qi_at_photons": qi_at_photons,
                "selected_indices": selected_indices,
                "skipped_indices": np.setdiff1d(np.arange(len(n_photons)), selected_indices),
                "model_qi": 1.0 / (slope * n_photons + intercept),
                "model_curve": {"n": n_model, "qi": model_qi},
                "success": True,
                "error_message": None
            }
            return results
        except Exception as e:
            # Handle errors
            results = {
                "error_message": f"Linear fit failed: {e}",
                "success": False
            }
            return results
    elif len(selected_indices) == 1:
        # For a single point, just return that value with defaults
        n_fit = n_photons[selected_indices[0]]
        qi_fit = qi_values[selected_indices[0]]
        
        # Return with default values
        results = {
            "fit_result": None,
            "params": {"QA": 0, "QTLS": 0, "nc": 0, "alpha": 0, "p1": 0, "p2": 0},
            "errors": {},
            "qi_at_photons": {1.0: qi_fit},  # Just return the single Qi
            "selected_indices": selected_indices,
            "skipped_indices": np.setdiff1d(np.arange(len(n_photons)), selected_indices),
            "model_qi": np.full_like(n_photons, qi_fit),
            "model_curve": {"n": np.array([n_fit]), "qi": np.array([qi_fit])},
            "success": True,
            "error_message": None
        }
        return results


# --- Plotting Functions ---

def plot_qc_vs_power(all_results, output_dir, base_filename):
    """Plots Qc vs log(Np) for all resonators on one graph."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for res_id, result in all_results.items():
        data = result['data']
        n_photons = data['n_photons']
        qc_values = data['qc_values']
        
        # Plot with different marker/color for each resonator
        ax.semilogx(n_photons, qc_values / 1e6, 'o-', label=f'Res {res_id}')
    
    ax.set_xlabel('Average Photon Number (<n>)')
    ax.set_ylabel('Coupling Q ($Q_c$) [Millions]')
    ax.set_title('$Q_c$ vs Power')
    ax.grid(True, which='both')
    ax.legend()
    plt.tight_layout()
    
    if output_dir:
        filepath = Path(output_dir) / f"{base_filename}_Qc.png"
        fig.savefig(filepath)
        print(f"Saved Qc plot: {filepath}")
    
    plt.show()
    plt.close(fig)

def plot_fo_vs_power(all_results, output_dir, base_filename):
    """Plots fo vs log(Np) for all resonators on one graph."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for res_id, result in all_results.items():
        data = result['data']
        n_photons = data['n_photons']
        fo_values = data['fo_values']
        
        # Plot with different marker/color for each resonator
        ax.semilogx(n_photons, fo_values / 1e9, 'o-', label=f'Res {res_id}')
    
    ax.set_xlabel('Average Photon Number (<n>)')
    ax.set_ylabel('Resonance Frequency ($f_0$) [GHz]')
    ax.set_title('$f_0$ vs Power')
    ax.grid(True, which='both')
    ax.legend()
    
    # Use offset format for y-axis if range is small
    ax.ticklabel_format(axis='y', useOffset=True, style='plain')
    plt.tight_layout()
    
    if output_dir:
        filepath = Path(output_dir) / f"{base_filename}_fo.png"
        fig.savefig(filepath)
        print(f"Saved fo plot: {filepath}")
    
    plt.show()
    plt.close(fig)

def plot_info(all_results, output_dir, base_filename):
    """Plots power levels and temperature data for all resonators."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    res_ids = list(all_results.keys())
    for i, res_id in enumerate(res_ids):
        result = all_results[res_id]
        data = result['data']
        n_photons = data['n_photons']
        temps = data['temps']
        
        # Plot photon number distribution for each resonator
        ax1.semilogy([i+1]*len(n_photons), n_photons, 'o')
        
        # Plot temperature for each resonator if available
        if not np.all(np.isnan(temps)):
            ax2.plot([i+1]*len(temps), temps, 's')
    
    # Set x-ticks to resonator IDs
    x_positions = np.arange(1, len(res_ids)+1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(res_ids)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(res_ids)
    
    ax1.set_xlabel('Resonator ID')
    ax1.set_ylabel('Average Photon Number (<n>)')
    ax1.set_title('Photon Number per Resonator')
    ax1.grid(True)
    
    ax2.set_xlabel('Resonator ID')
    ax2.set_ylabel('Temperature [K]')
    ax2.set_title('Temperature per Resonator')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        filepath = Path(output_dir) / f"{base_filename}_Info.png"
        fig.savefig(filepath)
        print(f"Saved info plot: {filepath}")
    
    plt.show()
    plt.close(fig)

def plot_qi_vs_power(all_results, output_dir, base_filename):
    """Plots Qi vs log(Np) and the fitted models for all resonators."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use different colors for different resonators
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for i, (res_id, result) in enumerate(all_results.items()):
        color = colors[i]
        data = result['data']
        n_photons = data['n_photons']
        qi_values = data['qi_values']
        
        # Plot data points
        sel_idx = result.get("selected_indices", np.array([]))
        if len(sel_idx) > 0:
            ax.semilogx(n_photons[sel_idx], qi_values[sel_idx]/1e6, 'o', 
                        color=color, 
                        label=f'{res_id}'
                        )
        
        # Plot skipped points with x marker
        skip_idx = result.get("skipped_indices", np.array([]))
        if len(skip_idx) > 0:
            ax.plot(n_photons[skip_idx], qi_values[skip_idx], 'x', color=color)
        
        # Plot fit model if successful
        if result["success"]:
            # Generate smooth curve for model
            n_min = max(1e-2, np.min(n_photons))
            n_max = np.max(n_photons)
            n_model = np.logspace(np.log10(n_min), np.log10(n_max), 100)
            
            params = result['params']
            qi_model = power_qi_model(n_model, params['QA'], params['QTLS'], 
                                      params['nc'], params['alpha'])
            
            ax.semilogx(n_model, qi_model/1e6, 
                        '-', color=color, 
                        # label=f'Res {res_id} (Fit)'
                        )
    
    ax.set_xlabel('Photon Number')
    ax.set_ylabel('$Q_i$ (1e6)')
    ax.set_title('$Q_i$ vs Power Fit')
    ax.grid(True, which='both')
    ax.legend()
    plt.tight_layout()
    
    if output_dir:
        filepath = Path(output_dir) / f"{base_filename}_QiFit.png"
        fig.savefig(filepath)
        print(f"Saved Qi fit plot: {filepath}")
    
    plt.show()
    plt.close(fig)

# --- Main Runner Function (Equivalent to RunPowerFit.m) ---

def run_power_fit(
    all_traces: List[ResonatorData],
    photon_values_for_qi1: List[float] = [1.0],
    group_by_key: str = 'resDesignator',
    output_dir: Optional[str] = None,
    base_filename: Optional[str] = None,
    skip_data_criteria: Optional[Dict[str, float]] = None, # e.g., {'iS21CircleFitRMSE': 0.1}
    generate_plots: bool = True,
    save_results: bool = True,
    power_weight_factor: float = 1.0,
) -> Dict[Any, Dict[str, Any]]:
    """
    Analyzes power dependence of resonators from a list of traces.

    Groups traces by a specified key, performs power-dependent Qi fits for each group,
    generates plots, and produces summary reports.

    Args:
        all_traces: List of ResonatorData objects, each representing a single trace
                    (measurement at one power level). Assumes each trace has fit
                    results (Qi, Qc, fo, Np) and metadata (temp, atten, resDesignator).
        photon_values_for_qi1: Photon numbers at which to calculate Qi (e.g., [1.0]).
        group_by_key: Attribute name in ResonatorData to group traces by resonator
                      (e.g., 'resDesignator', 'filename_base').
        output_dir: Directory to save plots and summary files. If None, uses the
                    directory of the first trace's data path.
        base_filename: Base name for output files. If None, attempts to create one
                       from the directory structure like the MATLAB script.
        skip_data_criteria: Dictionary of thresholds to skip individual traces
                            based on their fit quality metrics stored in ResonatorData.fit.gof.
                            Example: {'iS21CircleFitRMSE': 0.1, 'S21circSNR': 5}.
        generate_plots: If True, generate and save plots for Qc, fo, and Qi fits.
        save_results: If True, save the summary text files.

    Returns:
        Dictionary mapping each resonator ID (from group_by_key) to its
        power fit results dictionary returned by pd_fit, plus mean_temp and mean_fo.
    """
    if not all_traces:
        print("Warning: No traces provided to run_power_fit.")
        return {}

    ###################################### Saving Data ##########################################################

    # # --- Determine Output Path and Base Filename ---
    # first_trace = all_traces[0]
    # if output_dir is None:
    #     if hasattr(first_trace, 'save_path_nametag') and hasattr(first_trace, 'data_pathname'):
    #          # Try to replicate MATLAB's dated folder structure
    #          today = datetime.datetime.today().strftime('%Y%m%d')
    #          output_dir = Path(first_trace.data_pathname) / f"{first_trace.save_path_nametag}{today}"
    #     elif hasattr(first_trace, 'savePathname'): # From MATLAB structure?
    #          output_dir = Path(first_trace.savePathname)
    #     else:
    #          output_dir = Path('.') # Default to current directory
    # else:
    #     output_dir = Path(output_dir)

    # if base_filename is None:
    #     if hasattr(first_trace, 'base_filename'):
    #         base_filename = first_trace.base_filename # Use if pre-defined
    #     elif hasattr(first_trace, 'savePathname'): # Try to replicate MATLAB's logic
    #         try:
    #             parts = Path(first_trace.savePathname).parts
    #             if len(parts) >= 3:
    #                 base_filename = f"{parts[-3]} {parts[-2]}" # e.g., "ChipID CooldownID"
    #             else:
    #                 base_filename = output_dir.name # Fallback
    #         except Exception:
    #              base_filename = "power_fit_results" # Final fallback
    #     else:
    #         base_filename = "power_fit_results"

    # output_dir.mkdir(parents=True, exist_ok=True)
    # summary_filepath = output_dir / f"{base_filename}_Summary.txt"
    # pd_summary_filepath = output_dir / f"{base_filename}_PdFitSummary.txt"
    # print(f"Output directory: {output_dir}")
    # print(f"Base filename: {base_filename}")

    ################################################################################################

    # --- Group Traces by Resonator ID ---
    resonator_groups = defaultdict(list)
    for trace in all_traces:
        if not hasattr(trace, 'fit') or not isinstance(trace.fit, dict):
             print(f"Warning: Trace missing valid 'fit' dictionary. Skipping.")
             continue
        if not all(k in trace.fit for k in ['Qi', 'Qc', 'fo']) or not hasattr(trace, 'Np'):
             print(f"Warning: Trace missing essential fit results (Qi, Qc, fo) or Np. Skipping.")
             continue

        try:
            res_id = getattr(trace, group_by_key)
            resonator_groups[res_id].append(trace)
        except AttributeError:
            print(f"Warning: Trace missing group_by_key attribute '{group_by_key}'. Assigning to 'unknown'.")
            resonator_groups['unknown'].append(trace)

    all_power_fit_results = {}
    summary_lines = []
    pd_summary_lines = [
        f"PdFitSummary: {base_filename}",
        f"Date: {datetime.datetime.now()}",
        "---------------------------------------------------------"
    ]

    # --- Process Each Resonator Group ---
    for res_id, traces in resonator_groups.items():
        print(f"\nProcessing Resonator ID: {res_id} ({len(traces)} traces)")
        pd_summary_lines.append(f"\nResonator ID: {res_id}")

        # Sort traces by photon number for consistent analysis
        try:
            # Handle potential None or NaN Np values during sorting
            traces.sort(key=lambda r: float(r.Np) if r.Np is not None and not np.isnan(r.Np) else np.inf)
        except AttributeError:
             print(f"Warning: Cannot sort traces for resonator {res_id}. Missing 'Np'.")

        # --- Extract Data and Identify Traces to Skip ---
        n_photons_list, qi_values_list, qi_errors_list = [], [], []
        qc_values_list, fo_values_list, temps_list = [], [], []
        indices_to_skip = []

        for i, tr in enumerate(traces):
            # Store data regardless of skipping first
            n_photons_list.append(float(tr.Np))
            qi_values_list.append(float(tr.fit['Qi']))
            qi_errors_list.append(float(tr.fit.get('QiError', np.nan))) # Use .get for optional error
            qc_values_list.append(float(tr.fit['Qc']))
            fo_values_list.append(float(tr.fit['fo']))
            temps_list.append(float(getattr(tr, 'temp', np.nan)))

            # Check skip criteria for this trace
            skip_this = False
            if skip_data_criteria and hasattr(tr, 'fit') and isinstance(tr.fit.get('gof'), dict):
                gof = tr.fit['gof']
                for key, threshold in skip_data_criteria.items():
                    metric_value = gof.get(key)
                    if metric_value is not None:
                        # Assuming lower is better for RMSE, higher is better for SNR
                        is_rmse = 'rmse' in key.lower()
                        if is_rmse and metric_value > threshold:
                            skip_this = True
                            print(f"  Skipping trace {i} (Np={tr.Np:.2e}): {key}={metric_value:.3f} > {threshold:.3f}")
                            break
                        elif not is_rmse and metric_value < threshold:
                            skip_this = True
                            print(f"  Skipping trace {i} (Np={tr.Np:.2e}): {key}={metric_value:.3f} < {threshold:.3f}")
                            break
            if skip_this:
                indices_to_skip.append(i)

        # Convert lists to numpy arrays
        n_photons = np.array(n_photons_list, dtype=float)
        qi_values = np.array(qi_values_list, dtype=float)
        qi_errors = np.array(qi_errors_list, dtype=float)
        qc_values = np.array(qc_values_list, dtype=float)
        fo_values = np.array(fo_values_list, dtype=float)
        temps = np.array(temps_list, dtype=float)
        indices_to_skip = np.array(indices_to_skip, dtype=int)

        # Calculate means (excluding NaNs)
        mean_temp = np.nanmean(temps) if not np.all(np.isnan(temps)) else np.nan
        mean_fo = np.nanmean(fo_values) if not np.all(np.isnan(fo_values)) else np.nan

        # --- Perform Power Fit (PdFit equivalent) ---
        print(f"  Attempting power fit using {len(n_photons) - len(indices_to_skip)} points.")
        power_fit_result = pd_fit(n_photons, qi_values, qi_errors,
                                  photon_values_for_qi1=photon_values_for_qi1,
                                  skip_indices=indices_to_skip,
                                  power_weight_factor=power_weight_factor,)

        # Store results, including metadata
        power_fit_result['mean_temp'] = mean_temp
        power_fit_result['mean_fo'] = mean_fo
        power_fit_result['res_id'] = res_id # Store the ID for convenience
        # Store original data used (after initial NaN filtering but before skipping)
        power_fit_result['data'] = {
            'n_photons': n_photons,
            'qi_values': qi_values,
            'qi_errors': qi_errors,
            'qc_values': qc_values,
            'fo_values': fo_values,
            'temps': temps
        }
        all_power_fit_results[res_id] = power_fit_result

        # --- Add results to PdFit Summary ---
        if power_fit_result["success"]:
            params = power_fit_result['params']
            errors = power_fit_result['errors']
            qi1_dict = power_fit_result['qi_at_photons']
            qi1_str = ", ".join([f"Qi(n={n:.1f})={q:.2e}" for n, q in qi1_dict.items()])

            pd_summary_lines.append(f"  Fit Success: True")
            pd_summary_lines.append(f"  QA = {params['QA']:.3e} +/- {errors.get('QA', np.nan):.2e}")
            pd_summary_lines.append(f"  QTLS = {params['QTLS']:.3e} +/- {errors.get('QTLS', np.nan):.2e}")
            pd_summary_lines.append(f"  nc = {params['nc']:.3e} +/- {errors.get('nc', np.nan):.2e}")
            pd_summary_lines.append(f"  alpha = {params['alpha']:.3f} +/- {errors.get('alpha', np.nan):.2f}")
            pd_summary_lines.append(f"  {qi1_str}")
            if fit_result := power_fit_result.get("fit_result"):
                 pd_summary_lines.append(f"  Fit Chi-sqr: {fit_result.chisqr:.3e}, Reduced Chi-sqr: {fit_result.redchi:.3f}")
        else:
             pd_summary_lines.append(f"  Fit Success: False")
             pd_summary_lines.append(f"  Error: {power_fit_result.get('error_message', 'Unknown')}")

    # --- Generate Final Summary Report ---
    summary_lines.append(f"Loss Summary: {base_filename}")
    summary_lines.append(f"Date: {datetime.datetime.now()}")
    summary_lines.append("Fit Model: 1/Qi = 1/QA + (1/QTLS)*(1/(1+(n/nc)^alpha))")
    header = f"{'Res':>5s}  {'delta_A':>9s}  {'delta_TLS':>9s}  {'nc':>9s}  {'alpha':>6s}  {'Qi(n=1)':>9s}  {'<fo>(GHz)':>13s}  {'Temp(K)':>8s}  {'nMax':>9s}  {'max(Qi)':>9s}  {'nMin':>9s}  {'min(Qi)':>9s}"
    summary_lines.append(header)
    summary_lines.append("-" * len(header)) # Adjust length based on actual header rendering

    # Collect data for summary statistics
    stats_data = defaultdict(list)

    for res_id, result in all_power_fit_results.items():
         res_id_str = str(res_id)[:5] # Truncate ID if too long
         if result["success"]:
             params = result['params']
             tech_loss = 1.0 / params['QA'] if params['QA'] != 0 else np.inf
             tls_loss = 1.0 / params['QTLS'] if params['QTLS'] != 0 else np.inf
             nc = params['nc']
             alpha = params['alpha']
             n1qi = result['qi_at_photons'].get(1.0, np.nan)
             mean_fo_ghz = result['mean_fo'] / 1e9 if not np.isnan(result['mean_fo']) else np.nan
             mean_temp = result['mean_temp']

             # Get Qi/Np range from *selected* data used in the fit
             sel_idx = result['selected_indices']
             data = result['data']
             n_sel = data['n_photons'][sel_idx]
             qi_sel = data['qi_values'][sel_idx]

             n_max = np.max(n_sel) if len(n_sel) > 0 else np.nan
             qi_max = np.max(qi_sel) if len(qi_sel) > 0 else np.nan
             n_min = np.min(n_sel) if len(n_sel) > 0 else np.nan
             qi_min = np.min(qi_sel) if len(qi_sel) > 0 else np.nan

             summary_lines.append(
                 f"{res_id_str:>5s}  {tech_loss:9.2e}  {tls_loss:9.2e}  {nc:9.2e}  {alpha:6.3f}  "
                 f"{n1qi:9.2e}  {mean_fo_ghz:13.8f}  {mean_temp:8.3f}  "
                 f"{n_max:9.2e}  {qi_max:9.2e}  {n_min:9.2e}  {qi_min:9.2e}"
             )

             # Collect for stats (only add if not NaN/inf)
             if np.isfinite(tech_loss): stats_data['tech_loss'].append(tech_loss)
             if np.isfinite(tls_loss): stats_data['tls_loss'].append(tls_loss)
             if np.isfinite(nc): stats_data['nc'].append(nc)
             if np.isfinite(alpha): stats_data['alpha'].append(alpha)
             if np.isfinite(n1qi): stats_data['n1qi'].append(n1qi)
             if np.isfinite(mean_fo_ghz): stats_data['mean_fo'].append(mean_fo_ghz)
             if np.isfinite(mean_temp): stats_data['mean_temp'].append(mean_temp)
             # Add Qi ranges if needed for stats

         else:
             # Indicate fit failure in summary table
             summary_lines.append(
                 f"{res_id_str:>5s}  {'-- FAIL --':^69}" # Spans multiple columns
             )

    summary_lines.append("-" * len(header))

    # Calculate and add summary statistics (Mean, Min, Max, Std)
    if any(stats_data.values()): # Check if any successful fits occurred
        stats_labels = ['Max', 'Min', 'Avg', 'Std']
        # Use nan-aware functions
        stats_funcs = [np.nanmax, np.nanmin, np.nanmean, np.nanstd]
        stat_keys = ['tech_loss', 'tls_loss', 'nc', 'alpha', 'n1qi', 'mean_fo', 'mean_temp']
        formats = ['9.2e', '9.2e', '9.2e', '6.3f', '9.2e', '13.8f', '8.3f']

        for label, func in zip(stats_labels, stats_funcs):
            stat_values_str = []
            for key, fmt in zip(stat_keys, formats):
                arr = np.array(stats_data[key])
                if len(arr) > 0:
                    # Apply function, handle potential warnings for empty slices etc.
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)
                         val = func(arr)
                    stat_values_str.append(f"{val:{fmt}}" if np.isfinite(val) else f"{'N/A':>{len(fmt.split('.')[0])}}") # Format or N/A
                else:
                    stat_values_str.append(f"{'N/A':>{len(fmt.split('.')[0])}}") # N/A if no data

            # Construct line, matching header spacing
            line = f"{label:>5s}  {stat_values_str[0]}  {stat_values_str[1]}  {stat_values_str[2]}  "
            line += f"{stat_values_str[3]}  {stat_values_str[4]}  {stat_values_str[5]}  {stat_values_str[6]}  "
            # Add placeholders for Qi range stats if calculated
            line += f"{'':>9}  {'':>9}  {'':>9}  {'':>9}"
            summary_lines.append(line)

    # --- Generate Plots (moved outside the loop to plot all resonators together) ---
    if generate_plots:
        # plot_qc_vs_power(all_power_fit_results, output_dir, base_filename)
        # plot_fo_vs_power(all_power_fit_results, output_dir, base_filename)
        # plot_info(all_power_fit_results, output_dir, base_filename)
        plot_qi_vs_power(all_power_fit_results, output_dir, base_filename)

    # --- Write Summary Files ---
    # if save_results:
    #     try:
    #         with open(pd_summary_filepath, 'w') as f:
    #             f.write("\n".join(pd_summary_lines))
    #         print(f"Saved PdFit summary: {pd_summary_filepath}")
    #         with open(summary_filepath, 'w') as f:
    #             f.write("\n".join(summary_lines))
    #         print(f"Saved final summary: {summary_filepath}")
    #     except IOError as e:
    #         print(f"Error writing summary files to {output_dir}: {e}")
    print("\n".join(summary_lines)) # Print summary to console
    return all_power_fit_results



