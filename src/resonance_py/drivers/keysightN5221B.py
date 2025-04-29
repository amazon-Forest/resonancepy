from qcodes.instrument_drivers.Keysight import N52xx
import time
import numpy as np
from typing import TYPE_CHECKING, Any

from qcodes.instrument import VisaInstrument

from qcodes.parameters import (
    Parameter,
    ParameterBase,
    ParameterWithSetpoints,
    create_on_off_val_mapping,
)
from qcodes.validators import Numbers, Bool, Enum, Arrays

if TYPE_CHECKING:
    from typing_extensions import Unpack
    from collections.abc import Sequence
    from qcodes.instrument import VisaInstrumentKWArgs

# Define the SegmentedFrequencyAxis parameter
class FrequencyAxis(Parameter):
    """
    Parameter that returns the frequency points from the instrument.
    
    This parameter queries the instrument for the current frequency points
    along the x-axis, which is particularly useful for segmented sweeps where
    the frequency points may not be linearly spaced.
    
    Args:
        instrument: The parent KeysightN5221B instrument
        **kwargs: Additional Parameter arguments
    
    Returns:
        np.ndarray: Array of frequency points in Hz
    """
    def __init__(self, 
                 instrument: "KeysightN5221B", 
                 **kwargs: Any,
    ):
        super().__init__(instrument=instrument, **kwargs)

    def get_raw(self) -> np.ndarray:
        """
        Get the frequency points from the instrument.
        
        Returns:
            np.ndarray: Array of frequency points in Hz
        """
        root_instr = self.instrument

        data = root_instr.visa_handle.query_binary_values(
            "CALC:MEAS:DATA:X?", datatype="f", is_big_endian=True
            )
    
        return np.array(data)


class SegmentedSweep(N52xx.FormattedSweep):
    """
    Extension of FormattedSweep that supports segmented sweeps.
    
    This class enables running segmented sweeps and properly handling 
    the data and setpoints. It selects the appropriate axis parameter
    based on the current sweep type.
    
    Args:
        name: Parameter name
        instrument: Parent instrument
        sweep_format: Data format string (e.g., 'MLOG', 'PHAS')
        label: Parameter label
        unit: Parameter unit
        memory: Whether to use trace from memory
        **kwargs: Additional parameter arguments
    """

    def __init__(
        self,
        name: str,
        instrument: "KeysightN5221B",
        sweep_format: str,
        label: str,
        unit: str,
        memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, instrument, sweep_format, label, unit, memory, **kwargs)

    @property
    def setpoints(self) -> "Sequence[ParameterBase]":
        """
        Determine the appropriate setpoint parameter based on sweep type.
        
        For segmented sweeps, returns the frequency_axis parameter which
        contains the actual frequency points of the segmented sweep.
        
        Returns:
            Sequence[ParameterBase]: Tuple containing the appropriate axis parameter
            
        Raises:
            RuntimeError: If not attached to an instrument
            NotImplementedError: If sweep type is not supported
        """
        if self.instrument is None:
            raise RuntimeError("Cannot return setpoints if not attached to instrument")
        
        root_instrument = self.root_instrument  # type: ignore[assignment]
        sweep_type = root_instrument.sweep_type()
        
        if (sweep_type == "SEGM"):
            # For segmented sweep, we use the segmented_frequency_axis
            return (root_instrument.frequency_axis,)
        elif (sweep_type == "LIN"):
            return (root_instrument.frequency_axis,)
        elif (sweep_type == "LOG"):
            return (root_instrument.frequency_log_axis,)
        elif (sweep_type == "CW"):
            return (root_instrument.time_axis,)
        else:
            raise NotImplementedError(f"Axis for type {sweep_type} not implemented yet")

    @setpoints.setter
    def setpoints(self, setpoints: Any) -> None:
        """
        Stub to allow initialization. Ignores any set attempts on setpoint.
        
        Args:
            setpoints: Setpoints value to be ignored
        """
        return

    def get_raw(self) -> np.ndarray:
        """
        Get data for any sweep type including segmented sweeps.
        
        This method runs a sweep if auto_sweep is enabled and returns
        the formatted data from the instrument.
        
        Returns:
            np.ndarray: Measurement data
            
        Raises:
            RuntimeError: If not attached to an instrument
        """
        if self.instrument is None:
            raise RuntimeError("Cannot get data without instrument")
        
        root_instr = self.instrument.root_instrument
        # Check if we should run a new sweep
        auto_sweep = root_instr.auto_sweep()

        prev_mode = ""
        if auto_sweep:
            prev_mode = self.instrument.run_sweep()
            
        # Ask for data, setting the format to the requested form
        self.instrument.format(self.sweep_format)
        
        # Get the data (same for all sweep types)
        data = root_instr.visa_handle.query_binary_values(
            "CALC:DATA? FDATA", 
            datatype="f", 
            is_big_endian=True
            )
        data = np.array(data)
        

class KeysightN5221B(N52xx.KeysightPNABase):
    """
    QCoDeS driver for Keysight PNA N5221B Network Analyzer.
    
    This driver extends the base KeysightPNABase class with additional
    functionality, particularly for segmented sweeps. It provides methods
    to create, configure, and manipulate sweep segments, as well as
    perform various types of sweeps and retrieve complex S-parameter data.
    
    Args:
        name: Instrument name
        address: VISA resource address
        **kwargs: Additional visa instrument keyword arguments
    
    Examples:
        >>> pna = KeysightN5221B("pna", "TCPIP0::192.168.1.1::inst0::INSTR")
        >>> # Perform a linear sweep
        >>> data, freq = pna.linear_sweep(center_frequency=5e9, frequency_span=1e9)
        >>> # Create a segmented sweep focused on resonator frequencies
        >>> segments = pna.create_standard_segments([5.1e9, 6.3e9], fwhm_values=[2e6, 3e6])
        >>> data, freq = pna.segmented_sweep(segments)
    """
    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        """Initialize Keysight PNA N5221B driver."""
        super().__init__(
            name,
            address,
            min_freq=0,
            max_freq=13.51e9,
            min_power=-30,
            max_power=13,
            nports=4,
            **kwargs,
        )

        #------------------------------
        # Segments: parameters to configure segments 
        #------------------------------
        self.segment_add: Parameter = self.add_parameter(
            "segment_add",
            label="Add Segment",
            set_cmd="SENSe:SEGMent{}:ADD"
        )
        
        self.segment_count: Parameter = self.add_parameter(
            "segment_count",
            label="Count of Segments",
            get_cmd="SENSe:SEGMent:COUNT?"
        )
        
        self.segment_delete: Parameter = self.add_parameter(
            "segment_delete",
            label="Delete Segment",
            set_cmd="SENSe:SEGMent{}:DELETE"
        )

        self.segment_list: Parameter = self.add_parameter(
            "segment_list",
            label="List of Segments",
            get_cmd= self._get_segment_list,
            set_cmd=None,
        )

        #------------------------------
        # Axis: parameters to get frequency axis
        #------------------------------
        self.remove_parameter('frequency_axis')

        self.frequency_axis: FrequencyAxis = self.add_parameter(
            "frequency_axis",
            unit="Hz",
            label="Frequency",
            parameter_class=FrequencyAxis,
            # startparam=self.start,
            # stopparam=self.stop,
            # pointsparam=self.points,
            vals=Arrays(shape=(self.points,)),
        )
        for trace in self.traces:
            self._add_complex_parameter(trace)

        
    #-------------------------
    # Helper Methods
    #-------------------------
    def __get_or_default(self, value, default):
        """
        Return value if not None, otherwise return default.
        
        Args:
            value: Value to check
            default: Default value to return if value is None
            
        Returns:
            The value if not None, otherwise the default
        """
        return value if value is not None else default
    
    def _add_complex_parameter(self, trace):
        """
        Add a complex data parameter to a trace.
        
        This parameter returns the raw S-parameter data as complex numbers.
        
        Args:
            trace: The trace to add the parameter to
        """
        trace.add_parameter(
            "complex_data",
            sweep_format="SDAT",  # Use SDAT format for raw S-parameter data
            label="Complex S-Parameter",
            unit="",
            parameter_class=N52xx.FormattedSweep,
            get_parser=trace._parse_polar_data,  # Use the existing parser method
            vals=Arrays(shape=(self.points,), valid_types=(complex,)),
        )

    def get_complex_data(self):
        """
        Get complex S-parameter data directly from the active measurement.
        
        This method retrieves the raw S-parameter data as complex numbers,
        useful for further analysis of phase and amplitude information.
        
        Returns:
            np.ndarray: Complex S-parameter data array
        """
        data = self.visa_handle.query_binary_values(
            "CALC:DATA? SDATA", 
            datatype="f", 
            is_big_endian=True
        )
        # Convert alternating real/imaginary values to complex numbers
        data_array = np.array(data)
        complex_data = data_array[::2] + 1j * data_array[1::2]
        
        return complex_data
    
    def _parse_complex_data(self, data):
        """
        Parse the alternating real/imaginary values into complex numbers.
        
        Similar to the _parse_polar_data method in KeysightPNATrace.
        
        Args:
            data (np.ndarray): Raw data with alternating real/imaginary values
            
        Returns:
            np.ndarray: Array of complex numbers
        """
        # Reshape the flat array of alternating real/imaginary values into pairs
        real_imag = data.reshape((-1, 2))
        # Convert to complex numbers
        return real_imag[:, 0] + 1j * real_imag[:, 1]
    
    
    #--------------------------------------
    # Segments: Methods to configure segments 
    #--------------------------------------
    def _get_segment_list(self):
        """
        Get the list of segments from the instrument.
        
        Returns:
            np.ndarray: Array containing segment data
        """
        data = self.visa_handle.query_binary_values(
            "sense:segment:list? SSTOP", 
            datatype="f", 
            is_big_endian=True
        )
        return np.array(data)
    
    def segment_span(self, segment, freq: int = None):
        """
        Get or set the frequency span of a segment.
        
        Args:
            segment (int): Segment number
            freq (int, optional): Frequency span in Hz. If None, returns current span.
            
        Returns:
            str: Current span (if freq is None)
        """
        if freq is None:
            return self.ask(f"SENSe:SEGMent{segment}:FREQuency:SPAN?")
        else:
            self.write(f"SENSe:SEGMent{segment}:FREQuency:SPAN {freq}")
    
    def segment_delete_all(self):
        """Delete all segments from the instrument."""
        self.write(f"SENSe:SEGMent:DELete:ALL")
    
    def segment_center_frequency(self, segment, freq: float = None):
        """
        Get or set the center frequency of a segment.
        
        Args:
            segment (int): Segment number
            freq (float, optional): Center frequency in Hz. If None, returns current value.
            
        Returns:
            str: Current center frequency (if freq is None)
        """
        if freq is None:
            return self.ask(f"SENSe:SEGMent{segment}:FREQuency:CENTer?")
        else:
            self.write(f"SENSe:SEGMent{segment}:FREQuency:CENTer {freq}")
    
    def segment_start_frequency(self, segment, freq: float = None):
        """
        Get or set the start frequency of a segment.
        
        Args:
            segment (int): Segment number
            freq (float, optional): Start frequency in Hz. If None, returns current value.
            
        Returns:
            str: Current start frequency (if freq is None)
        """
        if freq is None:
            return self.ask(f"SENSe:SEGMent{segment}:FREQuency:START?")
        else:
            self.write(f"SENSe:SEGMent{segment}:FREQuency:START {freq}")
    
    def segment_stop_frequency(self, segment, freq: float = None):
        """
        Get or set the stop frequency of a segment.
        
        Args:
            segment (int): Segment number
            freq (float, optional): Stop frequency in Hz. If None, returns current value.
            
        Returns:
            str: Current stop frequency (if freq is None)
        """
        if freq is None:
            return self.ask(f"SENSe:SEGMent{segment}:FREQuency:STOP?")
        else:
            self.write(f"SENSe:SEGMent{segment}:FREQuency:STOP {freq}")

    def segment_points(self, segment, points: int = None):
        """
        Get or set the number of points in a segment.
        
        Args:
            segment (int): Segment number
            points (int, optional): Number of points. If None, returns current value.
            
        Returns:
            str: Current number of points (if points is None)
        """
        if points is None:
            return self.ask(f"SENSe:SEGMent{segment}:SWEep:POINts?")
        else:
            self.write(f"SENSe:SEGMent{segment}:SWEep:POINts {points}")
    
    def segment_if_bancwidth(self, segment, if_band: float = None):
        """
        Get or set the IF bandwidth of a segment.
        
        Args:
            segment (int): Segment number
            if_band (float, optional): IF bandwidth in Hz. If None, disables per-segment control.
        """
        if if_band is None:
            self.write(f'sense:segment{segment}:bandwidth:PORT:resolution:control OFF')
        else:
            self.write(f'sense:segment{segment}:bandwidth:PORT:resolution:control ON')
            self.write(f"SENS:SEGM{segment}:BWID {if_band}")
    
    def segment_state(self, segment:int, state: str = None):
        """
        Get or set the state (ON/OFF) of a segment.
        
        Args:
            segment (int): Segment number
            state (str, optional): State to set ('ON' or 'OFF'). If None, returns current state.
            
        Returns:
            str: Current state (if state is None)
            
        Raises:
            ValueError: If segment does not exist
        """
        if len(self.segment_list()) >= segment:
            if state is None:
                return self.ask(f"SENSe:SEGMent{segment}:STATE?")
            else:
                self.write(f"SENSe:SEGMent{segment}:STATE {state}")
        else:
            raise ValueError(f"Segment {segment} does not exist")

    def create_segments(self, 
                       segments_data: list, 
                       freq_mode: str = 'SSTOP'):
        """
        Configure multiple segments at once for a segmented frequency sweep.
        
        This method deletes all existing segments and creates new ones based on
        the provided segment data. It's an efficient way to set up a segmented sweep
        with a single command.
        
        Args:
            segments_data (list): List of dictionaries, each containing parameters for one segment:
                - state (str/int): Segment state ('ON'/'OFF' or 1/0)
                - points (int): Number of points in the segment
                - start/center (float): Start frequency (SSTOP) or center frequency (CSPAN) in Hz
                - stop/span (float): Stop frequency (SSTOP) or span (CSPAN) in Hz
                - ifbw (float, optional): IF bandwidth in Hz
                - dwell_time (float, optional): Dwell time in seconds
                - power (float, optional): Power level in dBm
            freq_mode (str): Frequency specification mode - 'SSTOP' (start/stop) or 'CSPAN' (center/span)
            
        Raises:
            ValueError: If no segment data is provided or required parameters are missing
            
        Example:
            >>> pna.create_segments([
            ...     {'state': 'ON', 'points': 201, 'start': 1e9, 'stop': 2e9, 'ifbw': 1e3},
            ...     {'state': 'ON', 'points': 401, 'start': 2.01e9, 'stop': 3e9, 'ifbw': 1e2}
            ... ])
        """
        if not segments_data:
            raise ValueError("No segment data provided")
        
        # First delete all existing segments to start fresh
        self.segment_delete_all()
        
        # Validate frequency mode
        if freq_mode not in ['SSTOP', 'CSPAN']:
            raise ValueError(f"Invalid frequency mode: {freq_mode}. Use 'SSTOP' or 'CSPAN'")
        
        # Prepare the list command
        num_segments = len(segments_data)
        segment_values = []
        
        # Create the segment data list
        for segment in segments_data:
            # Convert state to numeric value (1 for ON, 0 for OFF)
            if isinstance(segment.get('state'), str):
                state = 1 if segment.get('state', 'ON').upper() == 'ON' else 0
            else:
                state = 1 if segment.get('state', 1) else 0
            
            # Get required parameters
            points = segment.get('points', 201)
            
            # Get frequency values based on mode
            if freq_mode == 'SSTOP':
                start_freq = segment.get('start', None)
                stop_freq = segment.get('stop', None)
                
                if start_freq is None or stop_freq is None:
                    raise ValueError(f"Start and stop frequencies required in SSTOP mode")
                
                freq1, freq2 = start_freq, stop_freq
            else:  # CSPAN mode
                center_freq = segment.get('center', None)
                span_freq = segment.get('span', None)
                
                if center_freq is None or span_freq is None:
                    raise ValueError(f"Center and span frequencies required in CSPAN mode")
                
                freq1, freq2 = center_freq, span_freq
            
            # Get optional parameters with defaults
            ifbw = segment.get('ifbw', 1e3)  # Default to 1 kHz
            dwell_time = segment.get('dwell_time', 0)  # Default to 0 seconds
            power = segment.get('power', 0)  # Default to 0 dBm
            
            # Add segment values to the list
            segment_values.extend([state, points, freq1, freq2, ifbw, dwell_time, power])
        
        # Convert the segment values list to a comma-separated string
        segment_string = ','.join(str(val) for val in segment_values)
        
        # Send the command to create all segments at once
        self.write(f"SENSe:SEGMent:LIST {freq_mode},{num_segments},{segment_string}")
    
    def create_segment(self, segment_num=None, **kwargs):
        """
        Create a single segment with the given parameters.
        
        A convenience wrapper around create_segments for creating a single segment.
        
        Args:
            segment_num (int, optional): Segment number (will add to end if not specified)
            **kwargs: Parameters for the segment:
                - freq_mode (str): 'SSTOP' or 'CSPAN'
                - start/stop (float): Start/stop frequencies for SSTOP mode
                - center/span (float): Center/span for CSPAN mode
                - points (int): Number of points
                - ifbw (float): IF bandwidth
                - state (str/int): 'ON'/'OFF' or 1/0
            
        Raises:
            ValueError: If required parameters are missing
            
        Returns:
            dict: Information about the created segment
            
        Example:
            >>> pna.create_segment(
            ...     start=1e9, 
            ...     stop=2e9, 
            ...     points=201, 
            ...     ifbw=1e3
            ... )
        """
        # Check for required parameters
        freq_mode = kwargs.get('freq_mode', 'SSTOP')
        
        if freq_mode == 'SSTOP':
            if 'start' not in kwargs or 'stop' not in kwargs:
                raise ValueError("Missing required parameters: 'start' and 'stop' frequencies must be provided")
        elif freq_mode == 'CSPAN':
            if 'center' not in kwargs or 'span' not in kwargs:
                raise ValueError("Missing required parameters: 'center' and 'span' must be provided")
        
        # Create a list with a single segment dictionary
        self.create_segments([kwargs], freq_mode=freq_mode)
        
        # If segment_num was specified, enable that specific segment
        if segment_num is not None:
            self.segment_state(segment_num, 'ON')
            
        # Return information about the created segment for confirmation
        segment_count = int(self.segment_count())
        return {
            'segment_num': segment_num if segment_num is not None else segment_count,
            'parameters': kwargs,
            'status': 'created'
        }

    def perform_sweep(self, 
                     sweep_type='LIN',
                     center_frequency=None,
                     frequency_span=None, 
                     points=None,
                     if_bandwidth=None,
                     segments_data=None,
                     freq_mode='SSTOP', 
                     averaging=1,
                     measurement='S21',
                     format='MLOG'):
        """
        Perform a sweep of any type and return the data.
        
        This is the core measurement method that can perform linear, logarithmic,
        or segmented sweeps. It configures the instrument, runs the sweep, and
        returns both the complex data and frequency points.
        
        Args:
            sweep_type (str): Type of sweep - 'LIN', 'LOG', 'SEGM', etc.
            center_frequency (float, optional): Center frequency for linear/log sweeps (Hz)
            frequency_span (float, optional): Frequency span for linear/log sweeps (Hz)
            points (int, optional): Number of points for linear/log sweeps
            if_bandwidth (float, optional): IF bandwidth for linear/log sweeps (Hz)
            segments_data (list, optional): List of segment dictionaries for segmented sweep
            freq_mode (str): Frequency mode for segmented sweep - 'SSTOP' or 'CSPAN'
            averaging (int): Number of averages to perform (1 = no averaging)
            measurement (str): S-parameter to measure ('S11', 'S21', etc.)
            format (str): Data format ('MLOG', 'PHAS', 'SMITH', 'POLA', etc.)
            
        Returns:
            tuple: (complex_data, frequencies)
                - complex_data (np.ndarray): Complex S-parameter data
                - frequencies (np.ndarray): Frequency points in Hz
                
        Raises:
            ValueError: If segments_data is not provided for segmented sweep
            NotImplementedError: If sweep_type is not supported
            
        Example:
            >>> # Perform a linear sweep
            >>> data, freq = pna.perform_sweep(
            ...     sweep_type='LIN',
            ...     center_frequency=5e9,
            ...     frequency_span=1e9,
            ...     points=201,
            ...     if_bandwidth=1e3
            ... )
            >>> 
            >>> # Perform a segmented sweep
            >>> segments = [
            ...     {'state': 'ON', 'points': 201, 'start': 1e9, 'stop': 2e9},
            ...     {'state': 'ON', 'points': 401, 'start': 4e9, 'stop': 5e9}
            ... ]
            >>> data, freq = pna.perform_sweep(
            ...     sweep_type='SEGM',
            ...     segments_data=segments
            ... )
        """
        # Store original settings
        self.trace(measurement)
        self.format(format)
        original_averaging = self.averages_enabled()
        original_averaging_count = self.averages()
        original_sweep_type = self.sweep_type()
        
        # Configure sweep parameters based on sweep type
        if sweep_type in ['LIN', 'LOG']:
            # Linear or logarithmic sweep setup
            self.sweep_type(sweep_type)  # Set sweep type first
            if center_frequency is not None:
                self.center(center_frequency)
            if frequency_span is not None:
                self.span(frequency_span)
            if points is not None:
                self.points(points)
            if if_bandwidth is not None:
                self.if_bandwidth(if_bandwidth)
        elif sweep_type == 'SEGM':
            # Segmented sweep setup
            if segments_data:
                # Create segments before setting sweep type to ensure proper configuration
                self.create_segments(segments_data, freq_mode=freq_mode)
                # Explicitly set sweep type to segmented AFTER segments are defined
                self.sweep_type(sweep_type)
                
                # Verify segments were created properly
                segment_count = int(self.segment_count())
                if segment_count != len(segments_data):
                    self.log.warning(f"Expected {len(segments_data)} segments, but found {segment_count}")
                
                # Make sure the segments are active
                self.write("SENS:SWE:TYPE SEGM")  # Send direct command to ensure sweep type is segmented
            else:
                raise ValueError("segments_data must be provided for segmented sweep")
        else:
            # Other sweep types can be added here as needed
            raise NotImplementedError(f"Sweep type {sweep_type} not implemented yet")
        
        # Configure averaging
        if averaging > 1:
            self.averages_on()
            self.averages(averaging)
        else:
            self.averages_off()
        
        # Perform the sweep
        prev_mode = self.run_sweep()
        
        # For segmented sweep, wait slightly longer to ensure completion
        if sweep_type == 'SEGM':
            # Verify we're still in segmented mode
            current_type = self.sweep_type()
            if current_type != 'SEGM':
                self.log.warning(f"Sweep type changed to {current_type}, expected 'SEGM'")
                self.sweep_type('SEGM')  # Force it back
                self.run_sweep()  # Run another sweep to be safe
            
            # Give extra time for segmented sweep to complete
            total_points = sum(segment.get('points', 201) for segment in segments_data)
            time.sleep(0.01 * total_points)  # Add extra wait time proportional to number of points
        
        # Get the data
        frequencies = self.frequency_axis()
        time.sleep(1)
        complex_data = self.get_complex_data()

        #Restore original settings
        if averaging != original_averaging:
            self.averages_enabled(original_averaging)
            self.averages(original_averaging_count)
        self.sweep_mode(prev_mode)  # Restore previous sweep mode
        if original_sweep_type != sweep_type:
            self.sweep_type(original_sweep_type)  # Restore original sweep type
        
        return complex_data, frequencies

    def create_standard_segments(self, resonator_frequencies, fwhm_values=None, 
                               f_sec=None, n_points=None, 
                               ifbw_narrow=1e3, ifbw_wide=10e3):
        """
        Create a standard set of segments for resonator measurements.
        
        This method creates a segmented sweep optimized for measuring multiple
        resonators. It creates high-resolution segments near each resonator
        frequency and lower-resolution segments in between.
        
        Args:
            resonator_frequencies (list): List of resonator frequencies in Hz
            fwhm_values (list/float, optional): FWHM value(s) for scaling segments.
                Can be a single value or a list with one value per resonator.
                If None, defaults to 1% of each resonator's frequency.
            f_sec (list, optional): Factors of FWHM to define segment boundaries.
                Default is [0.5, 3, 30, 150].
            n_points (list, optional): Number of points for each segment category.
                Default is [75, 60, 40, 50].
            ifbw_narrow (float): IF bandwidth for segments near resonators (Hz)
            ifbw_wide (float): IF bandwidth for wide segments (Hz)
            
        Returns:
            list: List of segment dictionaries ready to use with segmented_sweep
            
        Example:
            >>> # Create segments for two resonators
            >>> segments = pna.create_standard_segments(
            ...     resonator_frequencies=[5.1e9, 6.3e9],
            ...     fwhm_values=[2e6, 3e6]
            ... )
            >>> # Use these segments in a sweep
            >>> data, freq = pna.segmented_sweep(segments)
        """
        # Import here to avoid circular imports
        from resonance_py.utils.segmentation import create_resonator_segments
        
        # If FWHM not provided, use a default of 1% of center frequency for each resonator
        if fwhm_values is None:
            fwhm_values = [freq * 0.01 for freq in resonator_frequencies]
            
        # Use default segment parameters if not specified
        if f_sec is None:
            f_sec = [0.5, 3, 30, 150]  # Similar to MATLAB example
        
        if n_points is None:
            n_points = [75, 60, 40, 50]  # Similar to MATLAB example
            
        # Create the segments using the external utility function
        segments = create_resonator_segments(
            resonator_frequencies=resonator_frequencies,
            fwhm_values=fwhm_values,
            f_sec=f_sec,
            n_points=n_points,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            ifbw_narrow=ifbw_narrow,
            ifbw_wide=ifbw_wide
        )
        
        return segments
    
    def linear_sweep(self, center_frequency=None, frequency_span=None, points=None, 
                   if_bandwidth=None, averaging=1, measurement='S21', format='MLOG'):
        """
        Perform a linear frequency sweep.
        
        Convenience wrapper for perform_sweep with linear sweep type.
        
        Args:
            center_frequency (float, optional): Center frequency in Hz
            frequency_span (float, optional): Frequency span in Hz
            points (int, optional): Number of points
            if_bandwidth (float, optional): IF bandwidth in Hz
            averaging (int): Number of averages (1 = no averaging)
            measurement (str): S-parameter to measure ('S11', 'S21', etc.)
            format (str): Data format ('MLOG', 'PHAS', etc.)
            
        Returns:
            tuple: (complex_data, frequencies)
                - complex_data (np.ndarray): Complex S-parameter data
                - frequencies (np.ndarray): Frequency points in Hz
                
        Example:
            >>> data, freq = pna.linear_sweep(
            ...     center_frequency=5e9,
            ...     frequency_span=1e9,
            ...     points=201,
            ...     if_bandwidth=1e3
            ... )
        """
        return self.perform_sweep(
            sweep_type='LIN',
            center_frequency=center_frequency,
            frequency_span=frequency_span,
            points=points,
            if_bandwidth=if_bandwidth,
            averaging=averaging,
            measurement=measurement,
            format=format
        )
    
    def segmented_sweep(self, segments_data, freq_mode='SSTOP', averaging=1, 
                       measurement='S21', format='MLOG'):
        """
        Perform a segmented frequency sweep.
        
        Convenience wrapper for perform_sweep with segmented sweep type.
        This allows for efficient measurement of multiple frequency ranges
        with different resolutions, which is ideal for measuring resonators.
        
        Args:
            segments_data (list): List of segment dictionaries
            freq_mode (str): Frequency specification mode - 'SSTOP' or 'CSPAN'
            averaging (int): Number of averages (1 = no averaging)
            measurement (str): S-parameter to measure ('S11', 'S21', etc.)
            format (str): Data format ('MLOG', 'PHAS', etc.)
            
        Returns:
            tuple: (complex_data, frequencies)
                - complex_data (np.ndarray): Complex S-parameter data
                - frequencies (np.ndarray): Frequency points in Hz
                
        Raises:
            ValueError: If no segment data is provided
            
        Example:
            >>> # Create segments using predefined method
            >>> segments = pna.create_standard_segments([5.1e9, 6.3e9])
            >>> # Or define segments manually
            >>> segments = [
            ...     {'state': 'ON', 'points': 201, 'start': 1e9, 'stop': 2e9},
            ...     {'state': 'ON', 'points': 401, 'start': 4e9, 'stop': 5e9}
            ... ]
            >>> # Perform the sweep
            >>> data, freq = pna.segmented_sweep(segments)
        """
        # Ensure segments are properly configured
        if not segments_data:
            raise ValueError("No segment data provided for segmented sweep")
        
        # Verify segments don't already exist or clear them
        if int(self.segment_count()) > 0:
            self.segment_delete_all()
        
        return self.perform_sweep(
            sweep_type='SEGM',
            segments_data=segments_data,
            freq_mode=freq_mode,
            averaging=averaging,
            measurement=measurement,
            format=format
        )


