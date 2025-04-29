import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional

def create_resonator_segments(
    resonator_frequencies: List[float],
    fwhm_values: Union[List[float], float],
    f_sec: List[float] = [0.5, 3, 30, 150],  # Factors of FWHM for different segments
    n_points: List[int] = [75, 60, 40, 50],  # Number of points for each segment
    min_freq: float = 10e6,
    max_freq: float = 13.1e9,
    ifbw_narrow: float = 1e3,
    ifbw_wide: float = 10e3,
    fill_gaps: bool = False,
    optimize: bool = False
) -> List[Dict[str, Any]]:
    """
    Create segments for resonator measurements similar to MATLAB LowPowerSpectrumV2.
    
    This function generates frequency segments for VNA measurements of resonators, with
    higher frequency resolution near resonant frequencies and lower resolution elsewhere.
    
    Args:
        resonator_frequencies: List of resonator center frequencies in Hz
        fwhm_values: Full Width at Half Maximum (FWHM) value(s) in Hz for scaling segments.
                     Can be a single value applied to all resonators or a list with one value per resonator.
        f_sec: Factors of FWHM to define segment boundaries. For example, the default
               [0.5, 3, 30, 150] creates segments that extend 0.5×FWHM, 3×FWHM, 30×FWHM,
               and 150×FWHM from each resonator's center frequency.
        n_points: Number of frequency points for each segment corresponding to f_sec boundaries.
                 The default [75, 60, 40, 50] assigns 75 points to the central segment,
                 60 points to the next segments, etc.
        min_freq: Minimum allowed frequency in Hz for any segment
        max_freq: Maximum allowed frequency in Hz for any segment
        ifbw_narrow: IF bandwidth in Hz for segments near resonators (higher resolution)
        ifbw_wide: IF bandwidth in Hz for segments far from resonators (lower resolution)
        fill_gaps: If True, adds segments to cover gaps between resonator segments and extends
                  to min_freq/max_freq boundaries
        optimize: If True, merges adjacent segments with identical measurement properties
                 to reduce total segment count
    
    Returns:
        List of segment dictionaries, each containing 'start', 'stop', 'points', and 'ifbw' keys,
        ready to use with VNA segmented sweep configuration
        
    Example:
        >>> freqs = [4.5e9, 5.2e9, 6.8e9]  # Three resonators at 4.5, 5.2, and 6.8 GHz
        >>> fwhm = 1e6  # 1 MHz FWHM for all resonators
        >>> segments = create_resonator_segments(freqs, fwhm, fill_gaps=True)
    """
    # Convert single FWHM value to a list if needed
    if isinstance(fwhm_values, (int, float)):
        fwhm_values = [fwhm_values] * len(resonator_frequencies)
    
    if len(fwhm_values) != len(resonator_frequencies):
        raise ValueError("fwhm_values must be a single value or a list matching resonator_frequencies")
    
    segments = []
    # Sort resonator frequencies to ensure segments are created in order
    resonator_frequencies = sorted(resonator_frequencies)
    
    # For each resonator, create segments in MATLAB style
    for i, (fo, fwhm) in enumerate(zip(resonator_frequencies, fwhm_values)):
        # MATLAB creates segments in this pattern:
        # 1. Outermost left segment
        # 2. Middle-outer left segment
        # 3. Middle-inner left segment
        # 4. Center segment (across resonator)
        # 5. Middle-inner right segment
        # 6. Middle-outer right segment
        # 7. Outermost right segment
        
        # Create each segment as in MATLAB, moving from outside in
        fo_segments = []
        
        # Create segments on both sides, starting from outermost
        for j in range(len(f_sec) - 1, 0, -1):
            # Left segment - distance from resonator decreases as j decreases
            fo_segments.append({
                'start': fo - f_sec[j] * fwhm,    # Outer boundary (further from resonator)
                'stop': fo - f_sec[j-1] * fwhm,   # Inner boundary (closer to resonator)
                'points': n_points[j],            # Number of frequency points for this segment
                'ifbw': ifbw_narrow if j < len(f_sec)-1 else ifbw_wide  # Use narrow IFBW near resonator
            })
        
        # Add center segment (contains the resonator frequency)
        fo_segments.append({
            'start': fo - f_sec[0] * fwhm,  # Left boundary of central segment
            'stop': fo + f_sec[0] * fwhm,   # Right boundary of central segment
            'points': n_points[0],          # Highest point density in central segment
            'ifbw': ifbw_narrow             # Narrowest IFBW for highest resolution
        })
        
        # Create segments on right side, moving outward
        for j in range(len(f_sec) - 1):
            # Right segment - distance from resonator increases as j increases
            fo_segments.append({
                'start': fo + f_sec[j] * fwhm,     # Inner boundary (closer to resonator)
                'stop': fo + f_sec[j+1] * fwhm,    # Outer boundary (further from resonator)
                'points': n_points[j+1],           # Number of frequency points for this segment
                'ifbw': ifbw_narrow if j < len(f_sec)-2 else ifbw_wide  # Use narrow IFBW near resonator
            })
        
        # Add these segments to our list
        segments.extend(fo_segments)
    
    # Sort all segments by starting frequency for consistent processing
    segments.sort(key=lambda x: x['start'])
    
    # Ensure frequency bounds don't exceed allowed limits
    for segment in segments:
        segment['start'] = max(min_freq, segment['start'])
        segment['stop'] = min(max_freq, segment['stop'])
    
    # Remove invalid segments (start >= stop or too narrow)
    # A minimum segment width of 1 kHz is enforced
    segments = [s for s in segments if s['stop'] > s['start'] and (s['stop'] - s['start']) >= 1e3]
    
    # Fix overlapping segments
    segments = _fix_overlapping_segments(segments)
    
    # Add segments to fill gaps if requested
    if fill_gaps:
        segments = _fill_segment_gaps(segments, min_freq, max_freq, n_points[-1], ifbw_wide)
    
    # Optimize segments if requested (merge adjacent segments with same properties)
    if optimize:
        segments = optimize_segments(segments)
    
    return segments


def _fix_overlapping_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix any overlapping segments by adjusting their boundaries to meet at the midpoint.
    
    This function ensures that no segments overlap in the frequency domain by
    adjusting the stop frequency of one segment and the start frequency of the
    next segment to meet at their midpoint.
    
    Args:
        segments: List of segment dictionaries, each with 'start' and 'stop' keys
                 representing frequency boundaries in Hz
        
    Returns:
        List of segment dictionaries with overlaps resolved
    
    Example:
        >>> segments = [{'start': 1e9, 'stop': 2e9}, {'start': 1.8e9, 'stop': 3e9}]
        >>> _fix_overlapping_segments(segments)
        [{'start': 1e9, 'stop': 1.9e9}, {'start': 1.9e9, 'stop': 3e9}]
    """
    if not segments:
        return segments
    
    # Sort segments by start frequency to ensure sequential processing
    segments = sorted(segments, key=lambda x: x['start'])
    
    # Fix any overlaps by adjusting to the midpoint
    for i in range(len(segments) - 1):
        if segments[i]['stop'] > segments[i+1]['start']:
            # Calculate midpoint between the overlapping boundaries
            mid = (segments[i]['stop'] + segments[i+1]['start']) / 2
            # Adjust both segments to meet at the midpoint
            segments[i]['stop'] = mid
            segments[i+1]['start'] = mid
    
    return segments


def _fill_segment_gaps(
    segments: List[Dict[str, Any]], 
    min_freq: float, 
    max_freq: float, 
    points: int, 
    ifbw: float
) -> List[Dict[str, Any]]:
    """
    Fill gaps between segments and at frequency extremes with additional segments.
    
    This ensures continuous frequency coverage across the entire specified range.
    New segments are created with the specified number of points and IF bandwidth.
    
    Args:
        segments: List of segment dictionaries, each with 'start' and 'stop' keys
        min_freq: Minimum frequency in Hz for the overall measurement range
        max_freq: Maximum frequency in Hz for the overall measurement range
        points: Number of frequency points to use for the fill segments
        ifbw: IF bandwidth in Hz to use for the fill segments
        
    Returns:
        Updated list of segments with gaps filled
    
    Example:
        >>> segments = [{'start': 2e9, 'stop': 3e9, 'points': 100, 'ifbw': 1e3}]
        >>> _fill_segment_gaps(segments, 1e9, 5e9, 50, 10e3)
        [
            {'start': 1e9, 'stop': 2e9, 'points': 50, 'ifbw': 10e3},
            {'start': 2e9, 'stop': 3e9, 'points': 100, 'ifbw': 1e3},
            {'start': 3e9, 'stop': 5e9, 'points': 50, 'ifbw': 10e3}
        ]
    """
    final_segments = []
    
    # Add segment from min_freq to first segment if needed
    if segments and segments[0]['start'] > min_freq:
        # Create a segment to cover from min_freq to the start of first existing segment
        final_segments.append({
            'start': min_freq,
            'stop': segments[0]['start'],
            'points': points,
            'ifbw': ifbw
        })
    
    # Add all existing segments and fill gaps between them
    for i, segment in enumerate(segments):
        final_segments.append(segment)
        
        # Add a segment between this and the next segment if there's a gap
        if i < len(segments) - 1 and segments[i+1]['start'] > segment['stop']:
            final_segments.append({
                'start': segment['stop'],
                'stop': segments[i+1]['start'],
                'points': points,
                'ifbw': ifbw
            })
    
    # Add segment from last segment to max_freq if needed
    if final_segments and final_segments[-1]['stop'] < max_freq:
        # Create a segment to cover from the end of last existing segment to max_freq
        final_segments.append({
            'start': final_segments[-1]['stop'],
            'stop': max_freq,
            'points': points,
            'ifbw': ifbw
        })
    
    return final_segments


def optimize_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optimize a list of segments by merging adjacent segments with identical measurement properties.
    
    This reduces the total number of segments while maintaining the same frequency coverage
    and measurement settings. Segments can be merged if they have the same number of points,
    IF bandwidth, and power settings (if specified).
    
    Args:
        segments: List of segment dictionaries with 'start', 'stop', 'points', 'ifbw', 
                 and optional 'power' keys
        
    Returns:
        Optimized list of segment dictionaries with redundant segments merged
    
    Example:
        >>> segments = [
        ...     {'start': 1e9, 'stop': 2e9, 'points': 50, 'ifbw': 10e3},
        ...     {'start': 2e9, 'stop': 3e9, 'points': 50, 'ifbw': 10e3},
        ...     {'start': 3e9, 'stop': 4e9, 'points': 100, 'ifbw': 1e3}
        ... ]
        >>> optimize_segments(segments)
        [
            {'start': 1e9, 'stop': 3e9, 'points': 50, 'ifbw': 10e3},
            {'start': 3e9, 'stop': 4e9, 'points': 100, 'ifbw': 1e3}
        ]
    """
    if not segments:
        return []
    
    # Sort segments by start frequency
    segments = sorted(segments, key=lambda x: x['start'])
    
    # Ensure no overlaps and resolve if found
    for i in range(len(segments) - 1):
        if segments[i]['stop'] > segments[i+1]['start']:
            # Calculate midpoint between overlapping boundaries
            mid = (segments[i]['stop'] + segments[i+1]['start']) / 2
            # Adjust both segments to meet at the midpoint
            segments[i]['stop'] = mid
            segments[i+1]['start'] = mid
    
    # Merge adjacent segments with same properties
    i = 0
    while i < len(segments) - 1:
        curr = segments[i]
        next_seg = segments[i+1]
        
        # Check if segments have same measurement properties
        same_properties = (
            curr['points'] == next_seg['points'] and
            curr.get('ifbw', 1e3) == next_seg.get('ifbw', 1e3) and
            curr.get('power', 0) == next_seg.get('power', 0)
        )
        
        # If same properties and frequencies match (or are very close), merge them
        # We use a 1 kHz tolerance to account for floating point imprecision
        if same_properties and abs(curr['stop'] - next_seg['start']) < 1e3:
            # Extend current segment to include the next segment
            curr['stop'] = next_seg['stop']
            # Remove the next segment as it's now merged
            segments.pop(i+1)
        else:
            # Move to next segment if we can't merge
            i += 1
    
    return segments


def calculate_expected_segments(
    resonator_frequencies: List[float],
    fwhm_values: Union[List[float], float],
    f_sec: List[float] = [0.5, 3, 30, 150],
    fill_gaps: bool = False
) -> int:
    """
    Calculate the expected number of segments based on the input parameters.
    
    This function estimates how many segments would be created by create_resonator_segments()
    given the input parameters. This is useful for verification or pre-allocating resources.
    
    Args:
        resonator_frequencies: List of resonator center frequencies in Hz
        fwhm_values: FWHM values for each resonator in Hz (single value or list)
        f_sec: Factors of FWHM defining segment boundaries around each resonator
        fill_gaps: Whether gaps between resonator segments will be filled
        
    Returns:
        Expected number of segments that would be created
    
    Example:
        >>> freqs = [4.5e9, 5.2e9, 6.8e9]  # Three resonators
        >>> calculate_expected_segments(freqs, 1e6)
        21  # 7 segments per resonator (with default f_sec=[0.5, 3, 30, 150])
        
        >>> calculate_expected_segments(freqs, 1e6, fill_gaps=True)
        25  # 21 resonator segments + up to 4 gap segments
    """
    # Each resonator gets 2*len(f_sec)-1 segments:
    # - Central segment: 1
    # - Left side segments: len(f_sec)-1 
    # - Right side segments: len(f_sec)-1
    segments_per_resonator = 2*len(f_sec) - 1
    num_resonator_segments = len(resonator_frequencies) * segments_per_resonator
    
    # If not filling gaps, that's all we have
    if not fill_gaps:
        return num_resonator_segments
    
    # With gap filling, we could have additional segments:
    # - 1 segment before the first resonator segment
    # - 1 segment after the last resonator segment
    # - Up to (len(resonator_frequencies)-1) segments between resonators
    # This is a maximum estimate - actual number depends on specific frequencies and FWHMs
    max_gap_segments = 2 + (len(resonator_frequencies) - 1)
    
    return num_resonator_segments + max_gap_segments

