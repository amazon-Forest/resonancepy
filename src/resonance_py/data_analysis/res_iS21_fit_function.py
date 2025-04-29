"""
Python port of the MATLAB res_iS21_fit_function.m
This function calculates the inverse S21 model with the same logic as the MATLAB code.
Includes lmfit implementation for fitting resonator data.
"""
import numpy as np
from lmfit import Model, Parameters


def res_is21_fit_function(Qi, Qc, phi, fo, A, B, theta, f):
    """
    Calculate inverse S21 response using the exact algorithm from MATLAB.
    
    Args:
        Qi: Internal quality factor (in millions)
        Qc: Coupling quality factor (in millions)
        phi: Coupling phase
        fo: Resonance frequency (in GHz)
        A: Amplitude scale
        B: Linear background (in 1e-3 units)
        theta: Rotation angle
        f: Frequency points (can be doubled for fitting compatibility)
    
    Returns:
        Flattened array of real and imaginary parts
    """
    # Using the freq range exactly as MATLAB does
    if len(f) % 2 == 0:
        freq = f[:len(f)//2]
    else:
        # For odd length arrays, MATLAB would still take half
        freq = f[:len(f)//2]
    
    # Initialize output array exactly as in MATLAB
    out = np.zeros((len(freq), 2))
    
    # Calculate normalized frequency deviation
    deltax = (freq - fo * 1e9) / (fo * 1e9)
    
    # Calculate the internal quality factor in natural units
    Qi_natural = Qi * 1e6
    
    # Port of the final formulas used in MATLAB
    f1 = 1 + (Qi/Qc) * (np.cos(phi) + 2*Qi_natural*deltax*np.sin(phi)) / (1 + (2*Qi_natural*deltax)**2)
    f2 = 0 + (Qi/Qc) * (np.sin(phi) - 2*Qi_natural*deltax*np.cos(phi)) / (1 + (2*Qi_natural*deltax)**2)
    
    # Including correction for the amplitude - element-wise division in MATLAB
    f1 = (1.0/A) * f1
    f2 = (1.0/A) * f2
    
    # Scale B as in MATLAB
    B = 1e3 * B
    
    # Calculate the output exactly as in MATLAB
    out[:, 0] = f1 * np.cos(-theta + B*deltax) - f2 * np.sin(-theta + B*deltax)
    out[:, 1] = f2 * np.cos(-theta + B*deltax) + f1 * np.sin(-theta + B*deltax)
    
    # Flatten exactly as MATLAB does
    out = out.flatten()
    
    # Handle odd length case exactly as in MATLAB
    if len(f) % 2 == 1:
        out = np.append(out, 0)
    
    return out


def resonator_model(f, Qi, Qc, phi, fo, A, B, theta):
    """
    Resonator model for lmfit - adapted to handle complex data.
    
    Args:
        f: Frequency array
        Qi, Qc, phi, fo, A, B, theta: Model parameters
    
    Returns:
        Complex S21 response for fitting
    """
    # Create frequency array for the model function
    f_doubled = np.tile(f, 2)
    
    # Get the real and imaginary parts
    result = res_is21_fit_function(Qi, Qc, phi, fo, A, B, theta, f_doubled)
    
    # Reshape the result to get real and imaginary parts
    real_part = result[:len(f)]
    imag_part = result[len(f):2*len(f)]
    
    # Return as complex array
    return real_part + 1j * imag_part


def fit_resonator_data(freq, s21_data):
    """
    Fit resonator data using lmfit and the inverse S21 model.
    
    Args:
        freq: Frequency points in Hz
        s21_data: Complex S21 data
    
    Returns:
        lmfit result object containing the fitted parameters
    """
    # Create real and imaginary components from data
    data_real = np.real(s21_data)
    data_imag = np.imag(s21_data)
    combined_data = np.concatenate((data_real, data_imag))
    
    # Create the model
    def is21_model(f, Qi, Qc, phi, fo, A, B, theta):
        # This wrapper handles the interface between our flattened model and lmfit
        return res_is21_fit_function(Qi, Qc, phi, fo, A, B, theta, f)
    
    # Create an lmfit model
    gmodel = Model(is21_model)
    
    # Create parameter object with initial values and bounds
    params = Parameters()
    
    # Calculate initial estimates for parameters
    f_center_idx = np.argmin(np.abs(s21_data))  # Find resonance frequency index
    f0_est = freq[f_center_idx] / 1e9  # Convert to GHz
    
    # Add parameters with initial values and reasonable bounds
    params.add('Qi', value=1.0, min=0.001, max=100)  # Internal Q in millions
    params.add('Qc', value=1.0, min=0.001, max=100)  # Coupling Q in millions
    params.add('phi', value=0.0, min=-np.pi, max=np.pi)  # Coupling phase
    params.add('fo', value=f0_est, min=f0_est*0.9, max=f0_est*1.1)  # Resonance freq in GHz
    params.add('A', value=1.0, min=0.1, max=10)  # Amplitude scale
    params.add('B', value=0.0, min=-10, max=10)  # Background in 1e-3 units
    params.add('theta', value=0.0, min=-np.pi, max=np.pi)  # Rotation angle
    
    # Fit the model
    f_doubled = np.tile(freq, 2)
    result = gmodel.fit(combined_data, params, f=f_doubled)
    
    return result


def plot_fit_result(freq, s21_data, result):
    """
    Plot the fitting results.
    
    Args:
        freq: Frequency points
        s21_data: Original S21 data
        result: lmfit result object
    """
    import matplotlib.pyplot as plt
    
    # Extract best fit parameters
    params = result.best_values
    
    # Generate the best fit
    f_doubled = np.tile(freq, 2)
    fit_data = res_is21_fit_function(
        params['Qi'], params['Qc'], params['phi'], 
        params['fo'], params['A'], params['B'], params['theta'], 
        f_doubled
    )
    
    # Reshape into complex data
    n = len(freq)
    fit_complex = fit_data[:n] + 1j * fit_data[n:2*n]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot real and imaginary components
    plt.subplot(221)
    plt.plot(freq, np.real(s21_data), 'b.', label='Data (Real)')
    plt.plot(freq, np.real(fit_complex), 'r-', label='Fit (Real)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Real(S21)')
    plt.legend()
    
    plt.subplot(222)
    plt.plot(freq, np.imag(s21_data), 'b.', label='Data (Imag)')
    plt.plot(freq, np.imag(fit_complex), 'r-', label='Fit (Imag)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Imag(S21)')
    plt.legend()
    
    # Plot amplitude and phase
    plt.subplot(223)
    plt.plot(freq, np.abs(s21_data), 'b.', label='Data (Magnitude)')
    plt.plot(freq, np.abs(fit_complex), 'r-', label='Fit (Magnitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|S21|')
    plt.legend()
    
    plt.subplot(224)
    plt.plot(freq, np.angle(s21_data), 'b.', label='Data (Phase)')
    plt.plot(freq, np.angle(fit_complex), 'r-', label='Fit (Phase)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase(S21) [rad]')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print fit results
    print("Fit Results:")
    print(f"Qi = {params['Qi']:.6f} million")
    print(f"Qc = {params['Qc']:.6f} million")
    print(f"phi = {params['phi']:.6f} rad")
    print(f"fo = {params['fo']:.9f} GHz")
    print(f"A = {params['A']:.6f}")
    print(f"B = {params['B']:.6f} (1e-3 units)")
    print(f"theta = {params['theta']:.6f} rad")
