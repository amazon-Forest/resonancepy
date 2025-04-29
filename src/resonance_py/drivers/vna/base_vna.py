import abc
import numpy as np
from time import sleep
from typing import Any, Dict, List, Tuple, Sequence, Union, Optional # Added Optional

# Assuming VisaInstrument and supporting types are imported correctly
# from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
# from qcodes.parameters import Parameter, ArrayParameter, MultiParameter, create_on_off_val_mapping # Adjusted imports
# from qcodes.validators import Enum, Numbers, Ints, Arrays

# Placeholder imports if qcodes is not fully available in this context
from qcodes.instrument import Instrument, VisaInstrument  # Use base Instrument for ABC example if VisaInstrument is complex
from qcodes.parameters import Parameter, ManualParameter # Use ManualParameter for placeholders where needed
from qcodes.validators import Enum, Numbers, Ints, Arrays


# --- Define Base VNA Class ---
class BaseVNA(Instrument, abc.ABC):
    """
    Abstract Base Class for Vector Network Analyzers.

    Defines a common interface and parameters expected for VNA drivers.
    Subclasses must implement the abstract methods to handle
    instrument-specific communication.
    """

    def __init__(
        self,
        name: str,
        **kwargs: Any, # Use Any if VisaInstrumentKWArgs not defined
    ) -> None:
        super().__init__(name, **kwargs)

        # --- Abstract Properties for Instrument Limits ---
        # Subclasses should define these, often in their __init__ before super call
        # or by overriding these properties.
        @property
        @abc.abstractmethod
        def min_freq(self) -> float:
            ...
        @property
        @abc.abstractmethod
        def max_freq(self) -> float:
            ...
        @property
        @abc.abstractmethod
        def min_power(self) -> float:
            ...
        @property
        @abc.abstractmethod
        def max_power(self) -> float:
            ...
        @property
        @abc.abstractmethod
        def nports(self) -> int:
            ...

        # --- Core Parameters ---
        self.add_parameter(
            "sweep_type",
            label="Sweep Type",
            get_cmd=self._get_sweep_type,
            set_cmd=self._set_sweep_type,
            vals=Enum("LIN", "LOG", "SEGM", "CW", "POW"), # Added POW from some VNAs
        )

        self.add_parameter(
            "center",
            label="Center Frequency",
            unit="Hz",
            get_cmd=self._get_center_freq,
            set_cmd=self._set_center_freq,
            # Validator uses the abstract properties defined above
            vals=Numbers(min_value=self.min_freq, max_value=self.max_freq),
        )

        self.add_parameter(
            "span",
            label="Frequency Span",
            unit="Hz",
            get_cmd=self._get_span,
            set_cmd=self._set_span,
            vals=Numbers(min_value=0, max_value=self.max_freq), # Span can be 0 for CW
        )

        self.add_parameter(
            "start_frequency",
            label="Start Frequency",
            unit="Hz",
            get_cmd=self._get_start_freq,
            set_cmd=self._set_start_freq,
            vals=Numbers(min_value=self.min_freq, max_value=self.max_freq),
        )

        self.add_parameter(
            "stop_frequency",
            label="Stop Frequency",
            unit="Hz",
            get_cmd=self._get_stop_freq,
            set_cmd=self._set_stop_freq,
            vals=Numbers(min_value=self.min_freq, max_value=self.max_freq),
        )

        self.add_parameter(
            "points",
            label="Sweep Points",
            unit="", # No unit for points
            get_cmd=self._get_points,
            set_cmd=self._set_points,
            vals=Ints(min_value=1, max_value=100001), # Generic wide range
        )

        self.add_parameter(
            "if_bandwidth",
            label="IF Bandwidth",
            unit="Hz",
            get_cmd=self._get_if_bandwidth,
            set_cmd=self._set_if_bandwidth,
            vals=Numbers(min_value=1, max_value=15e6), # Generic wide range
        )

        self.add_parameter(
            "power",
            label="Source Power",
            unit="dBm",
            get_cmd=self._get_power,
            set_cmd=self._set_power,
            vals=Numbers(min_value=self.min_power, max_value=self.max_power),
        )

        # --- Averaging Parameters ---
        self.add_parameter(
            "averages_enabled",
            label="Averaging ON/OFF",
            get_cmd=self._get_averages_enabled,
            set_cmd=self._set_averages_enabled,
            # Using create_on_off_val_mapping for standard boolean mapping
            # val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            val_mapping={True: 1, False: 0}, # Simpler version
        )

        self.add_parameter(
            "averages",
            label="Averaging Count",
            unit="",
            get_cmd=self._get_averages,
            set_cmd=self._set_averages,
            vals=Ints(min_value=1, max_value=65536), # Generic wide range
        )

        # --- Sweep Control ---
        self.add_parameter(
            "auto_sweep",
            label="Continuous Sweep",
            get_cmd=self._get_auto_sweep,
            set_cmd=self._set_auto_sweep,
            # val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
            val_mapping={True: 1, False: 0}, # Simpler version
        )

        # --- Trace / Measurement Format ---
        self.add_parameter(
            "format",
            label="Display Format",
            get_cmd=self._get_format,
            set_cmd=self._set_format,
            # Common formats - subclasses might override vals if different
            vals=Enum(
                "MLOG", "PHAS", "SMIT", "POL", "GDEL", "MLIN", "SWR", "REAL", "IMAG",
                "UPH", "PPH", # Some VNAs have unfolded/positive phase
                # Keysight specific from example (may not be universal)
                "SLIN", "SLOG", "SCOM", "SADM", "PLIN", "PLOG",
            ),
        )

        # Note: The 'trace' parameter from the example is complex.
        # It acts as a getter for count and setter for selection.
        # It's often better to have separate methods/parameters in a base class.
        # We will define abstract methods for selecting trace and getting count.

        # Note: 'data_format' (ASC, REAL, REAL32) is often tied to specific
        # data transfer commands. Let's handle data fetching via abstract methods.

        # --- High-level helper parameters (optional in base) ---
        # FrequencyAxis and complex_data handling often involve custom Parameter
        # classes. Base class can define abstract *methods* to get this data.
        # Subclasses can then implement these methods, potentially by adding
        # their own specialized Parameter instances internally.

        # --- Segment Sweep Parameters (Minimal Interface) ---
        # Full segment table manipulation can be complex and VNA-specific.
        # Provide a basic interface for enabling segment sweep and configuring.
        # self.segment_add: Parameter = self.add_parameter(...) # etc.
        # For simplicity in BaseVNA, we might only include an abstract method
        # to configure segments from a list/dict structure.

    # ==================================================================
    # Abstract Methods - Subclasses MUST Implement These
    # ==================================================================

    # --- Communication for Core Parameters ---
    @abc.abstractmethod
    def _get_sweep_type(self) -> str: ...
    @abc.abstractmethod
    def _set_sweep_type(self, value: str) -> None: ...

    @abc.abstractmethod
    def _get_center_freq(self) -> float: ...
    @abc.abstractmethod
    def _set_center_freq(self, value: float) -> None: ...

    @abc.abstractmethod
    def _get_span(self) -> float: ...
    @abc.abstractmethod
    def _set_span(self, value: float) -> None: ...

    @abc.abstractmethod
    def _get_start_freq(self) -> float: ...
    @abc.abstractmethod
    def _set_start_freq(self, value: float) -> None: ...

    @abc.abstractmethod
    def _get_stop_freq(self) -> float: ...
    @abc.abstractmethod
    def _set_stop_freq(self, value: float) -> None: ...

    @abc.abstractmethod
    def _get_points(self) -> int: ...
    @abc.abstractmethod
    def _set_points(self, value: int) -> None: ...

    @abc.abstractmethod
    def _get_if_bandwidth(self) -> float: ...
    @abc.abstractmethod
    def _set_if_bandwidth(self, value: float) -> None: ...

    @abc.abstractmethod
    def _get_power(self) -> float: ...
    @abc.abstractmethod
    def _set_power(self, value: float) -> None: ...

    # --- Communication for Averaging ---
    @abc.abstractmethod
    def _get_averages_enabled(self) -> bool: ... # Should return mapped value (True/False)
    @abc.abstractmethod
    def _set_averages_enabled(self, value: bool) -> None: ... # Takes mapped value (True/False)

    @abc.abstractmethod
    def _get_averages(self) -> int: ...
    @abc.abstractmethod
    def _set_averages(self, value: int) -> None: ...

    # --- Communication for Sweep Control ---
    @abc.abstractmethod
    def _get_auto_sweep(self) -> bool: ... # Should return mapped value (True/False)
    @abc.abstractmethod
    def _set_auto_sweep(self, value: bool) -> None: ... # Takes mapped value (True/False)

    # --- Communication for Trace / Measurement ---
    @abc.abstractmethod
    def _get_format(self) -> str: ...
    @abc.abstractmethod
    def _set_format(self, value: str) -> None: ...

    @abc.abstractmethod
    def _select_trace(self, trace_num: int) -> None:
        """Selects the active trace/measurement number for subsequent configuration."""
        ...

    @abc.abstractmethod
    def _get_selected_trace(self) -> Optional[int]:
        """Gets the currently selected trace number, if applicable."""
        ...

    @abc.abstractmethod
    def _define_measurement(self, trace_num: int, measurement: str) -> None:
        """Defines the measurement (e.g., 'S21', 'S11') for a given trace."""
        ...

    @abc.abstractmethod
    def _get_measurement_definition(self, trace_num: int) -> Optional[str]:
        """Gets the measurement definition (e.g., 'S21') for a given trace."""
        ...

    # --- Data Acquisition ---
    @abc.abstractmethod
    def _get_frequency_axis(self) -> np.ndarray:
        """Returns the frequency points for the current sweep setup."""
        ...

    @abc.abstractmethod
    def _get_complex_data(self, trace_num: int) -> np.ndarray:
        """
        Fetches the complex measurement data (e.g., S-parameters) for the specified trace.
        Should return a numpy array of complex numbers.
        """
        ...

    # --- Segment Sweep (Basic Interface) ---
    @abc.abstractmethod
    def _create_segments(self, segments_data: List[Dict[str, Any]]) -> None:
        """
        Configures the instrument for a segmented sweep based on the provided data.
        Subclasses will implement the specific command format.
        """
        ...

    # --- Sweep Execution ---
    @abc.abstractmethod
    def _trigger_sweep(self) -> None:
        """Initiates a single sweep or set of averages."""
        ...

    @abc.abstractmethod
    def _wait_for_sweep_completion(self, timeout: Optional[float] = None) -> None:
        """Blocks until the current sweep operation is complete."""
        ...

    # ==================================================================
    # Optional High-Level Methods (Common Logic)
    # ==================================================================
    # These methods can often be implemented in the base class using the
    # public parameter interface and abstract methods defined above.

    def run_sweep(self, wait: bool = True) -> None:
        """
        Puts the VNA in single sweep mode (if necessary), triggers a sweep,
        and optionally waits for completion.

        Args:
            wait: If True, waits for the sweep to complete before returning.
        """
        # Store previous mode if we need to restore it later
        prev_mode_continuous = self.auto_sweep()

        if prev_mode_continuous:
            self.auto_sweep(False)
            # Add a small delay if needed for the instrument to settle
            sleep(0.1) # Adjust as necessary

        self._trigger_sweep()

        if wait:
            # Use a sensible timeout based on sweep settings if possible,
            # otherwise rely on the abstract method's internal logic or default.
            estimated_time = self.ask(":SENS:SWE:TIME?").strip() # Example SCPI
            timeout = None
            try:
                # Add buffer to sweep time
                timeout = float(estimated_time) * (self.averages.get_latest() or 1) + 2.0
            except:
                timeout = self.timeout # Fallback to default timeout

            self._wait_for_sweep_completion(timeout=timeout)

        # Optionally restore continuous mode - often desired to leave it in single sweep
        # if prev_mode_continuous:
        #     self.auto_sweep(True)

    def perform_sweep(
        self,
        sweep_type: str = "LIN",
        *,
        start_frequency: Optional[float] = None,
        stop_frequency: Optional[float] = None,
        center_frequency: Optional[float] = None,
        frequency_span: Optional[float] = None,
        points: Optional[int] = None,
        if_bandwidth: Optional[float] = None,
        power: Optional[float] = None,
        segments_data: Optional[List[Dict[str, Any]]] = None,
        averaging: int = 1,
        trace_num: int = 1, # Default to trace 1
        measurement: str = "S21", # Default S-parameter
        format: str = "MLOG", # Default display format
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Configures the VNA for a specific sweep, performs it, and returns data.

        Handles setting parameters, triggering, and fetching data.
        Restores previous averaging state.

        Returns:
            tuple[np.ndarray, np.ndarray]: (complex_data, frequency_axis)
        """
        # Backup relevant settings
        old_avg_state = self.averages_enabled()
        old_avg_cnt = self.averages()
        old_type = self.sweep_type()
        # Potentially backup more state if needed (e.g., format, selected trace)

        try:
            # Configure measurement
            self._select_trace(trace_num)
            self._define_measurement(trace_num, measurement)
            self.format(format) # Set display format (may affect data if not getting SDAT)

            # Configure sweep parameters
            self.sweep_type(sweep_type)

            if sweep_type in {"LIN", "LOG", "CW"}:
                # Prioritize start/stop over center/span if both potentially provided
                if start_frequency is not None:
                    self.start_frequency(start_frequency)
                elif center_frequency is not None:
                    self.center(center_frequency)

                if stop_frequency is not None:
                    self.stop_frequency(stop_frequency)
                elif frequency_span is not None:
                    self.span(frequency_span)

                if points is not None:
                    self.points(points)
            elif sweep_type == "SEGM":
                if segments_data is None:
                    raise ValueError("segments_data is required for segmented sweep")
                self._create_segments(segments_data)
            else:
                # POW sweep might need different params
                 raise NotImplementedError(f"Sweep type {sweep_type} configuration not fully implemented in base perform_sweep")

            # Set common parameters
            if if_bandwidth is not None:
                self.if_bandwidth(if_bandwidth)
            if power is not None:
                self.power(power)

            # Configure averaging
            avg_enabled = averaging > 1
            self.averages_enabled(avg_enabled)
            if avg_enabled:
                self.averages(averaging)

            # Run sweep and get data
            self.run_sweep(wait=True) # Ensure sweep completes

            # Fetch data - use the abstract methods
            freq_axis = self._get_frequency_axis()
            complex_data = self._get_complex_data(trace_num) # Get data for the configured trace

            return complex_data, freq_axis

        finally:
            # Restore previous settings
            self.averages_enabled(old_avg_state)
            self.averages(old_avg_cnt)
            self.sweep_type(old_type)
            # Restore other settings if necessary

    # Convenience wrappers can also live in the base class
    def linear_sweep(
        self,
        start_frequency: Optional[float] = None,
        stop_frequency: Optional[float] = None,
        points: Optional[int] = None,
        if_bandwidth: Optional[float] = None,
        power: Optional[float] = None,
        averaging: int = 1,
        trace_num: int = 1,
        measurement: str = "S21",
        format: str = "MLOG",
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.perform_sweep(
            sweep_type="LIN",
            start_frequency=start_frequency,
            stop_frequency=stop_frequency,
            points=points,
            if_bandwidth=if_bandwidth,
            power=power,
            averaging=averaging,
            trace_num=trace_num,
            measurement=measurement,
            format=format,
        )

    def segmented_sweep(
        self,
        segments_data: List[Dict[str, Any]],
        *,
        power: Optional[float] = None, # IFBW/Power might be per-segment
        averaging: int = 1,
        trace_num: int = 1,
        measurement: str = "S21",
        format: str = "MLOG",
    ) -> Tuple[np.ndarray, np.ndarray]:
         return self.perform_sweep(
            sweep_type="SEGM",
            segments_data=segments_data,
            power=power, # Apply power globally if set here
            averaging=averaging,
            trace_num=trace_num,
            measurement=measurement,
            format=format,
        )

# --- Level 2: Base for VISA-based VNAs ---
class VisaBaseVNA(BaseVNA, VisaInstrument):
    """
    Intermediate Base Class for VNAs communicating via VISA.
    Inherits the VNA interface and adds VISA capabilities.
    """
    def __init__(
        self,
        name: str,
        address: str,
        terminator: str = '\n',
        timeout: int = 5,
        **kwargs # Pass remaining kwargs to Instrument/VnaInterface
    ) -> None:
        # Initialize VnaInterface first (doesn't need address)
        # Pass only relevant kwargs if needed, or pass all
        BaseVNA.__init__(self, name, **kwargs)
        # Initialize VisaInstrument
        VisaInstrument.__init__(self, name, address, terminator=terminator, timeout=timeout, **kwargs)
        # Note: name is passed twice, QCoDeS handles this okay.

    # Optional: Provide concrete implementation for VERY common VISA commands
    # def _trigger_sweep(self) -> None:
    #     self.write("INIT:IMM") # Example

    # Most abstract methods from VnaInterface remain abstract here,
    # to be implemented by the final concrete VISA driver.

# Example of how a subclass would use it (Conceptual)
class ConcreteVNA(BaseVNA):

    # Subclass MUST define these properties
    # @property
    # def min_freq(self) -> float: return 10e3
    @property
    def max_freq(self) -> float: return 9e9
    @property
    def min_power(self) -> float: return -60
    @property
    def max_power(self) -> float: return 10
    @property
    def nports(self) -> int: return 2 # Example

    def __init__(self, name: str, address: str, **kwargs):
        # Note: In a real scenario, you might set properties *before* super().__init__
        # if the base class __init__ directly uses them (e.g., in validators).
        # Or, define them as class attributes if they are static.
        super().__init__(name, address, **kwargs)
        # Add any instrument-specific parameters or initial setup here
        self.log.info(f"Connected to {self.name} (Concrete Implementation)")
         # Example: Add frequency axis parameter specific to this implementation
        # self.add_parameter('frequency',
        #                    label='Frequency Axis',
        #                    unit='Hz',
        #                    get_cmd=self._get_frequency_axis_concrete,
        #                    vals=Arrays(shape=(self.points.get_latest,)))
    # --- Implement ALL abstract methods from BaseVNA ---

    def _get_sweep_type(self) -> str: return self.ask("SENS:SWE:TYPE?")
    def _set_sweep_type(self, value: str) -> None: self.write(f"SENS:SWE:TYPE {value}")

    def _get_center_freq(self) -> float: return float(self.ask("SENS:FREQ:CENT?"))
    def _set_center_freq(self, value: float) -> None: self.write(f"SENS:FREQ:CENT {value:.4E}")

    def _get_span(self) -> float: return float(self.ask("SENS:FREQ:SPAN?"))
    def _set_span(self, value: float) -> None: self.write(f"SENS:FREQ:SPAN {value:.4E}")

    def _get_start_freq(self) -> float: return float(self.ask("SENS:FREQ:STAR?"))
    def _set_start_freq(self, value: float) -> None: self.write(f"SENS:FREQ:STAR {value:.4E}")

    def _get_stop_freq(self) -> float: return float(self.ask("SENS:FREQ:STOP?"))
    def _set_stop_freq(self, value: float) -> None: self.write(f"SENS:FREQ:STOP {value:.4E}")

    def _get_points(self) -> int: return int(self.ask("SENS:SWE:POIN?"))
    def _set_points(self, value: int) -> None: self.write(f"SENS:SWE:POIN {value}")

    def _get_if_bandwidth(self) -> float: return float(self.ask("SENS:BAND:RES?"))
    def _set_if_bandwidth(self, value: float) -> None: self.write(f"SENS:BAND:RES {value:.4E}")

    def _get_power(self) -> float: return float(self.ask("SOUR:POW?")) # Example SCPI
    def _set_power(self, value: float) -> None: self.write(f"SOUR:POW {value:.2f}")

    def _get_averages_enabled(self) -> bool: return int(self.ask("SENS:AVER:STAT?")) == 1
    def _set_averages_enabled(self, value: bool) -> None: self.write(f"SENS:AVER:STAT {1 if value else 0}")

    def _get_averages(self) -> int: return int(self.ask("SENS:AVER:COUN?"))
    def _set_averages(self, value: int) -> None: self.write(f"SENS:AVER:COUN {value}")

    def _get_auto_sweep(self) -> bool: return int(self.ask("INIT:CONT?")) == 1
    def _set_auto_sweep(self, value: bool) -> None: self.write(f"INIT:CONT {1 if value else 0}")

    def _get_format(self) -> str: return self.ask("CALC:FORM?").strip()
    def _set_format(self, value: str) -> None: self.write(f"CALC:FORM {value}")

    def _select_trace(self, trace_num: int) -> None: self.write(f":CALC:PAR{trace_num}:SEL")
    def _get_selected_trace(self) -> Optional[int]:
        # This is tricky - SCPI might not have a direct query for selected trace num
        # May need internal state tracking or parsing CALC:PAR:CAT?
        self.log.warning("Getting selected trace might not be directly supported.")
        return None # Placeholder

    def _define_measurement(self, trace_num: int, measurement: str) -> None: self.write(f"CALC{trace_num}:PAR:DEF {measurement}") # Corrected SCPI example
    def _get_measurement_definition(self, trace_num: int) -> Optional[str]: return self.ask(f"CALC{trace_num}:PAR:DEF?").strip()

    def _get_frequency_axis(self) -> np.ndarray:
        # Fetch frequency axis data - SCPI varies ('CALC:X?', 'SENS:FREQ:DATA?')
        raw_freq_data = self.ask("SENS:FREQ:DATA?") # Example SCPI
        return np.fromstring(raw_freq_data, sep=',')

    def _get_complex_data(self, trace_num: int) -> np.ndarray:
        # Select the trace first (important!)
        self._select_trace(trace_num)
        # Use appropriate command and parsing for complex data (SDATA is common)
        # Need to ensure VNA is set to output ASCII for this method
        # self.write("FORM:DATA ASC,0") # Example: Ensure ASCII output
        raw = self.query_ascii_values(f"CALC{trace_num}:DATA:SDAT?") # Corrected SCPI example
        data = np.asarray(raw)
        if len(data) == 0:
             self.log.warning(f"No complex data returned for trace {trace_num}")
             num_points = self.points.get_latest() or 1
             return np.full(num_points, np.nan + 1j*np.nan) # Return NaNs matching expected size

        # Data is typically interleaved real, imag
        return data[::2] + 1j * data[1::2]

    def _create_segments(self, segments_data: List[Dict[str, Any]]) -> None:
        # Implementation specific to the Keysight E5072A example format
        if not segments_data:
            raise ValueError("No segment data provided")
        num_segments = len(segments_data)
        # Header format for E5072A might be different - check manual
        # This is based on the provided E5072A example structure:
        cmd_parts = ["5", "0", "0", "0", "0", "0", str(num_segments)]
        for segment in segments_data:
            start_freq = float(segment["start"])
            stop_freq = float(segment["stop"])
            points = int(segment.get("points", 201))
            # E5072A SENS:SEGM:DATA format might not include IFBW/Power/Dwell per segment directly in this command
            cmd_parts.extend([str(start_freq), str(stop_freq), str(points)])
        cmd_string = ",".join(cmd_parts)
        self.write(f"SENS:SEGM:DATA {cmd_string}")
        sleep(0.5) # Allow processing time

    def _trigger_sweep(self) -> None:
        self.write("INIT:IMM")

    def _wait_for_sweep_completion(self, timeout: Optional[float] = None) -> None:
        # Use OPC (Operation Complete) query - common method
        # Timeout handling should be managed by the VISA layer ideally
        # self.visa_handle.query("*OPC?", read_termination='\n', timeout=timeout) # Example with pyvisa timeout
        _ = self.ask("*OPC?") # Simpler ask, relies on underlying timeout
        # Fallback/alternative: Poll status byte or condition register
        # start_time = time.time()
        # while True:
        #     try:
        #         # Check status condition register - specific bits indicate sweeping
        #         # Example: :STAT:OPER:COND? often has bit 4 or 5 for sweep
        #         condition = int(self.ask(":STAT:OPER:COND?").strip())
        #         if not (condition & (1 << 4)): # Check if sweep bit (e.g., bit 4) is 0
        #              break
        #     except Exception as e:
        #         self.log.warning(f"Error polling status: {e}")
        #         # Fallback to simple wait based on sweep time? Risky.
        #         break # Avoid infinite loop on error
        #     if timeout is not None and (time.time() - start_time > timeout):
        #         raise TimeoutError("Sweep completion polling timed out")
        #     sleep(0.05) # Poll interval

# --- Example Usage ---
# if __name__ == '__main__':
#     try:
#         # You cannot instantiate BaseVNA directly:
#         # base_vna = BaseVNA("base_test", "GPIB0::1::INSTR") # Raises TypeError

#         # Instantiate the concrete implementation
#         concrete_vna = ConcreteVNA("my_concrete_vna", "GPIB0::1::INSTR", timeout=10)

#         print("\n--- Testing Parameters ---")
#         print(f"Center Freq: {concrete_vna.center()} Hz")
#         concrete_vna.center(2.5e9)
#         print(f"Points: {concrete_vna.points()}")
#         concrete_vna.points(401)
#         print(f"IF Bandwidth: {concrete_vna.if_bandwidth()} Hz")
#         concrete_vna.averages_enabled(True)
#         concrete_vna.averages(4)
#         print(f"Averaging: {concrete_vna.averages_enabled()}, Count: {concrete_vna.averages()}")

#         print("\n--- Performing Linear Sweep ---")
#         data, freq = concrete_vna.linear_sweep(
#             start_frequency=1e9,
#             stop_frequency=3e9,
#             points=101,
#             if_bandwidth=1e3,
#             averaging=2,
#             measurement="S21"
#         )
#         print(f"Frequency Axis Shape: {freq.shape}")
#         print(f"Data Shape: {data.shape}")
#         print(f"First data point: {data[0]}")
#         print(f"Last data point: {data[-1]}")

#         # print("\n--- Performing Segmented Sweep ---")
#         # segments = [
#         #     {'start': 1e9, 'stop': 1.1e9, 'points': 51},
#         #     {'start': 2e9, 'stop': 2.1e9, 'points': 51, 'ifbw': 500}, # Example IFBW override (if supported by _create_segments impl.)
#         # ]
#         # seg_data, seg_freq = concrete_vna.segmented_sweep(segments_data=segments)
#         # print(f"Segmented Frequency Axis Shape: {seg_freq.shape}")
#         # print(f"Segmented Data Shape: {seg_data.shape}")


#     except TypeError as e:
#         print(f"\nINSTANTIATION ERROR: {e}")
#         print("This is expected if trying to instantiate BaseVNA or if a subclass misses abstract methods.")
#     except Exception as e:
#         print(f"\nRUNTIME ERROR: {e}")
#     finally:
#         # Clean up instances
#         if 'concrete_vna' in locals():
#             concrete_vna.close()
#         # Instrument.close_all() # Close all qcodes instruments