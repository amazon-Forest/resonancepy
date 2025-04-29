import numpy as np

def s21_curve(f, fr, Qi, Qc_mag, phi, mode='notch'):
    re_inv_Qc = np.cos(phi) / Qc_mag
    inv_Ql = (1.0 / Qi) + re_inv_Qc
    Ql = 1.0 / inv_Ql
    Qc_complex = Qc_mag * np.exp(-1j * phi)
    # Calculate the S21 parameter

    if mode == 'notch':
        S21 = 1 - (Ql / Qc_complex) / (1 + 2j * Ql * (f - fr) / fr)
    elif mode == 'reflection':
        S21 = 1 - (2* Ql / Qc_complex) / (1 + 2j * Ql * (f - fr) / fr)
    elif mode == 'inline':
        S21 = (Ql / Qc_complex) / (1 + 2j * Ql * (f - fr) / fr)
    
    return S21


def generate_multi_resonator_s21(
    freq: np.ndarray,
    resonance_frequencies: list,
    Qi_values: list = None,
    Qc_mag_values: list = None,
    phi_values: list = None,
    noise_level: float = 0.001,
    baseline_ripple: float = 0.05,
    baseline_period: float = 2.0,
):
    """
    Generate simulated S21 data with multiple resonators and realistic noise.
    
    Parameters:
    -----------
    resonance_frequencies : list
        List of resonance frequencies for each resonator (in Hz)
    Qi_values : list, optional
        List of internal quality factors for each resonator. 
        If None, defaults to 1e6 for all resonators.
    Qc_mag_values : list, optional
        List of coupling quality factor magnitudes for each resonator.
        If None, defaults to 2e5 for all resonators.
    phi_values : list, optional
        List of coupling angles (in radians) for each resonator.
        If None, defaults to -0.1 for all resonators.
    noise_level : float, optional
        Level of Gaussian noise to add (standard deviation)
    baseline_ripple : float, optional
        Amplitude of sinusoidal ripple in the baseline
    baseline_period : float, optional
        Period of the baseline ripple relative to the frequency span
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Complex S21 data and frequency array
    """
    # Get frequency axis
    num_points = len(freq)
    
    # Set default quality factor values if not provided
    n_resonators = len(resonance_frequencies)
    if Qi_values is None:
        Qi_values = [1e6] * n_resonators
    if Qc_mag_values is None:
        Qc_mag_values = [2e5] * n_resonators
    if phi_values is None:
        phi_values = [-0.1] * n_resonators
        
    # Ensure all parameter lists have the same length
    if not (len(Qi_values) == len(Qc_mag_values) == len(phi_values) == n_resonators):
        raise ValueError("All parameter lists must have the same length as resonance_frequencies")
    
    # Start with unity transmission (S21 = 1)
    s21_total = np.ones(num_points, dtype=complex)
    
    # Add each resonator's response (multiply for series configuration)
    for i in range(n_resonators):
        # Generate S21 for this resonator
        s21_res = s21_curve(
            f=freq, 
            fr=resonance_frequencies[i], 
            Qi=Qi_values[i], 
            Qc_mag=Qc_mag_values[i], 
            phi=phi_values[i]
        )
        
        # Multiply by the total S21 (series resonators)
        s21_total *= s21_res
    
    # Add sinusoidal baseline ripple (common in real VNA measurements)
    freq_span = freq[-1] - freq[0]
    ripple_freq = baseline_period * (2 * np.pi / freq_span)
    baseline = 1 + baseline_ripple * np.sin(ripple_freq * (freq - freq[0]))
    phase_ripple = baseline_ripple * 0.2 * np.cos(ripple_freq * 1.3 * (freq - freq[0]))
    
    # Apply baseline effects
    s21_total = s21_total * baseline * np.exp(1j * phase_ripple)
    
    # Add Gaussian noise to both real and imaginary parts
    noise_real = np.random.normal(0, noise_level, num_points)
    noise_imag = np.random.normal(0, noise_level, num_points)
    s21_total += (noise_real + 1j * noise_imag)
    
    return s21_total

