import numpy as np

def comp2magdB(complex_data):
    """
    Convert complex data to magnitude in dB
    
    Args:
        complex_data: Complex data array
        
    Returns:
        Magnitude in dB
    """
    return 20 * np.log10(np.abs(complex_data))

def normalize_data(data):
    avg = np.average(data)
    std = np.std(data)
    #peaks_indx, peaks = find_peaks(data, height=[max_height,max_height]) #height=[min_val,max_height]

    normalized_data_ = ( (data - avg) / std )
    normalized_data = normalized_data_ - np.min(normalized_data_)

    return normalized_data


def stats_of_data(data):
    """
    Calculate statistics of a data array
    
    Args:
        data: Input data array
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }