from __future__ import annotations

"""
QCoDeS driver for the Keysight **E5072A / E5071B** ENA‑series RF network‑analyser.

This file is an almost drop‑in replacement for :class:`KeysightN5221B`.
The public API (attributes, parameter names, helper‑methods) is kept **identical**
so that existing measurement notebooks keep working.  The only substantive
changes are:

* The class inherits directly from :class:`qcodes.instrument.VisaInstrument`
  (the ENA does not use :class:`N52xx.KeysightPNABase`).
* All SCPI commands are updated to the E507x command set that was extracted
  from the Programmer's Guide (9th ed., firmware A.08.10) citeturn4file10.

If you used the old PNA driver you should be able to do::

    from keysightE5072A import KeysightE5072A as KeysightVNA

    vna = KeysightVNA("ena", "TCPIP0::192.168.0.10::inst0::INSTR")
    data, freq = vna.linear_sweep(center_frequency=5e9, frequency_span=1e9)

and everything will behave the same – only faster, because we removed layers of
legacy abstraction :-)
"""

from time import sleep
from typing import Any, Sequence, TYPE_CHECKING, Dict, List, Union

import numpy as np
from qcodes.instrument import VisaInstrument
from qcodes.parameters import (
    Parameter,
    ParameterBase,
    ParameterWithSetpoints,
)
from qcodes.validators import Arrays, Numbers, Bool, Enum, Ints

if TYPE_CHECKING:
    from typing_extensions import Unpack
    from qcodes.instrument import VisaInstrumentKWArgs

# -----------------------------------------------------------------------------
# Helper parameters
# -----------------------------------------------------------------------------
class FrequencyAxis(Parameter):
    """Return the stimulus axis for the active channel/trace (in Hz)."""
    def __init__(self, 
                 instrument: "KeysightE5072A", 
                 **kwargs: Any,
    ):
        super().__init__(instrument=instrument, **kwargs)

    def get_raw(self) -> np.ndarray:  # type: ignore[override]
        root_instrument = self.instrument  # type: ignore[attr-defined]
        data = root_instrument.visa_handle.query_ascii_values(":SENS:FREQ:DATA?")
        return np.asarray(data)


class _FormattedSweep(ParameterWithSetpoints):
    """Generic formatted sweep parameter that respects *auto_sweep*."""

    def __init__(
        self,
        name: str,
        instrument: "KeysightE5072A",
        sweep_format: str,
        label: str,
        unit: str,
        memory: bool = False,
        **kwargs: Any,
    ) -> None:
        self.sweep_format = sweep_format.upper()
        self._memory = memory
        super().__init__(name=name, instrument=instrument, label=label, unit=unit, **kwargs)

    # ------------------------------------------------------------------
    # setpoint handling – identical public contract as original driver
    # ------------------------------------------------------------------
    @property  # type: ignore[override]
    def setpoints(self) -> Sequence[ParameterBase]:  # noqa: D401
        sweep_type = self.root_instrument.sweep_type()
        if sweep_type == "SEGM":
            return (self.root_instrument.frequency_axis,)  # type: ignore[attr-defined]
        else:
            return (self.root_instrument.frequency_axis,)  # linear / log use same axis

    @setpoints.setter  # type: ignore[override]
    def setpoints(self, _: Any) -> None:  # noqa: D401 – keep silent setter
        return

    # ------------------------------------------------------------------
    # data acquisition
    # ------------------------------------------------------------------
    def get_raw(self) -> np.ndarray:  # type: ignore[override]
        root = self.root_instrument  # type: ignore[attr-defined]
        if root.auto_sweep():
            root.run_sweep()
        root.format(self.sweep_format)
        raw = root.visa_handle.query_binary_values(
            "CALC:DATA? FDATA", datatype="f", is_big_endian=True
        )
        return np.asarray(raw)


# -----------------------------------------------------------------------------
# Main instrument class – PUBLIC API identical to KeysightN5221B
# -----------------------------------------------------------------------------
class KeysightE5072A(VisaInstrument):
    """QCoDeS driver replicating *KeysightN5221B* for the ENA‑series."""

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)

        # basic identity information -------------------------------------------------
        idn = self.IDN()
        self.log.info("Connected to %s – FW %s", idn.get("model", "?"), idn.get("firmware", "?"))

        # user‑accessible limits (same attr‑names as PNA driver)
        self.min_freq: float = 9e3  # ENA‑10 kHz typically
        self.max_freq: float = 8.5e9  # E5071B upper limit – adjust for option
        self.min_power: float = -55
        self.max_power: float = +10
        self.nports: int = 4


        # ------------------------------------------------------------------
        # core low‑level parameters (names kept identical)
        # ------------------------------------------------------------------
        self.add_parameter(
            "sweep_type",
            label="Sweep Type",
            set_cmd="SENS:SWE:TYPE {0}",
            get_cmd="SENS:SWE:TYPE?",
            vals=Enum("LIN", "LOG", "SEGM", "CW"),
        )

        self.add_parameter(
            "center",
            label="Center Frequency",
            unit="Hz",
            set_cmd="SENS:FREQ:CENT {0}",
            get_cmd="SENS:FREQ:CENT?",
            vals=Numbers(self.min_freq, self.max_freq),
        )

        self.add_parameter(
            "span",
            label="Frequency Span",
            unit="Hz",
            set_cmd="SENS:FREQ:SPAN {0}",
            get_cmd="SENS:FREQ:SPAN?",
            vals=Numbers(0, self.max_freq),
        )

        self.add_parameter(
            "points",
            label="Sweep Points",
            set_cmd="SENS:SWE:POIN {0}",
            get_cmd="SENS:SWE:POIN?",
            vals=Ints(1, 16001),
        )

        self.add_parameter(
            "if_bandwidth",
            label="IF Bandwidth",
            unit="Hz",
            set_cmd="SENS:BAND:RES {0}",
            get_cmd="SENS:BAND:RES?",
            vals=Numbers(1, 3e6),
        )

        # averaging ------------------------------------------------------------------
        self.add_parameter(
            "averages_enabled",
            label="Averaging ON/OFF",
            set_cmd="SENS:AVER:STAT {0}",
            get_cmd="SENS:AVER:STAT?",
            val_mapping={True: 1, False: 0},
        )

        self.add_parameter(
            "averages",
            label="Averaging Count",
            set_cmd="SENS:AVER:COUN {0}",
            get_cmd="SENS:AVER:COUN?",
            vals=Ints(1, 65535),
        )

        # auto/hold sweep -------------------------------------------------------------
        self.add_parameter(
            "auto_sweep",
            label="Continuous Sweep",
            set_cmd="INIT:CONT {0}",
            get_cmd="INIT:CONT?",
            val_mapping={True: 1, False: 0},
        )

        # active trace / measurement --------------------------------------------------
        self.add_parameter(
            "trace",
            label="Active Measurement",
            set_cmd=":CALC:PAR{}:SEL",
            get_cmd= ":CALC:PAR:COUN?",
        )

        self.add_parameter(
            "format",
            label="Display Format",
            set_cmd="CALC:FORM {0}",
            get_cmd="CALC:FORM?",
            vals=Enum(
                "MLOG", "PHAS", "SMIT", "POL", "GDEL", "MLIN", "SWR", "REAL", "IMAG",
                "SLIN", "SLOG", "SCOM", "SADM", "PLIN", "PLOG",
            ),
        )

        self.add_parameter(
            "data_format",
            label="Data Output Format",
            set_cmd="CALC:FORM {0}",
            get_cmd="CALC:FORM?",
            vals=Enum(
                "ASC", "REAL", "REAL32",
            ),
        )

        # high‑level helper parameters ------------------------------------------------
        # (their *names* must match the legacy driver)
        # self.remove_parameter("frequency_axis", raise_if_missing=False)
        self.add_parameter(
            "frequency_axis",
            unit="Hz",
            label="Frequency",
            parameter_class=FrequencyAxis,
        )

        # complex data helper (kept identical) ----------------------------------------
        # self.add_parameter(
        #     "complex_data",
        #     label="Complex S‑parameter",
        #     parameter_class=_FormattedSweep,
        #     sweep_format="SDAT",
        #     unit="",
        #     vals=Arrays(shape=(self.points,)),  # type: ignore[arg-type]
        # )

        # ------------------------------------------------------------------
        # segment‑table helpers (identical names)
        # ------------------------------------------------------------------
        self.segment_add: Parameter = self.add_parameter(
            "segment_add", label="Add Segment", set_cmd="SENS:SEGM:ADD")
        self.segment_count: Parameter = self.add_parameter(
            "segment_count", label="Segment Count", get_cmd="SENS:SEGM:COUN?")
        self.segment_delete: Parameter = self.add_parameter(
            "segment_delete", label="Delete Segment", set_cmd="SENS:SEGM:DEL {0}")
        self.segment_list: Parameter = self.add_parameter(
            "segment_list", label="List Segments", get_cmd=self._get_segment_list)

    # ==================================================================
    # Low‑level helpers (names identical)
    # ==================================================================
    def __get_or_default(self, value: Any, default: Any) -> Any:  # noqa: D401
        return value if value is not None else default


    # ------------------ complex data helper ---------------------------
    def get_complex_data(self) -> np.ndarray:  # identical name
        raw = self.visa_handle.query_ascii_values("CALC:DATA:SDATA?")
        data = np.asarray(raw)
        return data[::2] + 1j * data[1::2]

    # ------------------ segment table helpers -------------------------
    def _get_segment_list(self) -> np.ndarray:
        data = self.visa_handle.query_binary_values("SENS:SEGM:LIST?", datatype="f", is_big_endian=True)
        return np.asarray(data)

    def segment_start_frequency(self, seg: int, freq: float | None = None):
        if freq is None:
            return float(self.ask(f"SENS:SEGM{seg}:FREQ:STAR?"))
        self.write(f"SENS:SEGM{seg}:FREQ:STAR {freq}")

    def segment_stop_frequency(self, seg: int, freq: float | None = None):
        if freq is None:
            return float(self.ask(f"SENS:SEGM{seg}:FREQ:STOP?"))
        self.write(f"SENS:SEGM{seg}:FREQ:STOP {freq}")

    def segment_points(self, seg: int, pts: int | None = None):
        if pts is None:
            return int(self.ask(f"SENS:SEGM{seg}:SWE:POIN?"))
        self.write(f"SENS:SEGM{seg}:SWE:POIN {pts}")

    def segment_if_bancwidth(self, seg: int, ifbw: float | None = None):
        if ifbw is None:
            return float(self.ask(f"SENS:SEGM{seg}:BWID?"))
        self.write(f"SENS:SEGM{seg}:BWID {ifbw}")

    def segment_state(self, seg: int, state: str | None = None):
        if state is None:
            return self.ask(f"SENS:SEGM{seg}:STAT?").strip()
        state = state.upper()
        assert state in {"ON", "OFF", "1", "0"}
        self.write(f"SENS:SEGM{seg}:STAT {state}")


    # ------------------ bulk segment creation -------------------------
    def create_segments(self, segments_data: List[Dict[str, Any]]):
        """
        Configure multiple segments at once for a segmented frequency sweep.
        
        This method deletes all existing segments and creates new ones based on
        the provided segment data. It follows the E5072A's SCPI command syntax.
        
        Args:
            segments_data (list): List of dictionaries, each containing parameters for one segment:
                - state (str/int): Segment state ('ON'/'OFF' or 1/0) - will be ignored as E5072A format doesn't use it
                - points (int): Number of points in the segment
                - start/center (float): Start frequency in Hz
                - stop/span (float): Stop frequency in Hz
                - ifbw (float, optional): IF bandwidth in Hz (default: 1e3)
                - power (float, optional): Power level in dBm (default: 0)
                - dwell (float, optional): Dwell time in seconds (default: 0)
            freq_mode (str): Frequency mode - 'SSTOP' only supported for E5072A
            
        Raises:
            ValueError: If no segment data is provided or required parameters are missing
        """
        if not segments_data:
            raise ValueError("No segment data provided")
        
        # Prepare the segment data command
        num_segments = len(segments_data)
        cmd_parts = ["5", "0", "0", "0", "0", "0", str(num_segments)]
    
        # Add each segment's parameters
        for segment in segments_data:
            # Get required parameters with validation
            try:
                start_freq = float(segment["start"])
                stop_freq = float(segment["stop"])
                points = int(segment.get("points", 201))
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid segment data: {e}") from e
            
            # Get optional parameters with defaults
            ifbw = segment.get("ifbw", 1e3)  # Default to 1 kHz
            power = segment.get("power", 0)  # Default to 0 dBm
            dwell = segment.get("dwell", 0)  # Default to 0 seconds
            time_per_point = segment.get("time", 0)  # Default to 0 (auto)
            
            # Add parameters for this segment
            # <star>,<stop>,<nop>,<ifbw>,<pow>,<del>,<time>
            cmd_parts.extend([
                str(start_freq),
                str(stop_freq),
                str(points),
                # str(ifbw),
                # str(power),
                # str(dwell),
                # str(time_per_point)
            ])

        # Combine all parts into a single command
        cmd_string = ",".join(cmd_parts)

        cmd_string
        # Send the command to create all segments at once
        self.write(f"SENS:SEGM:DATA {cmd_string}")
        sleep(0.5)  # Allow time for the command to be processed




    # ==================================================================
    # High‑level measurement helpers (names identical)
    # ==================================================================
    # def run_sweep(self):
    #     """Trigger a single sweep and wait until it completes (†)."""
    #     # store current mode            
    #     prev = bool(int(self.ask("INIT:CONT?")))
    #     if prev:
    #         self.auto_sweep(False)
    #     self.write("INIT:IMM")
    #     self.visa_handle.query("*OPC?")  # block until complete
    #     if prev:
    #         self.auto_sweep(True)
    #     return prev
    def run_sweep(self) -> str:
        """
        Run a set of sweeps on the network analyzer.
        Note that this will run all traces on the current channel.
        """
        # Store previous mode
        prev_mode = prev = bool(int(self.ask("INIT:CONT?"))) # On Hold or Continuous
        
        self.auto_sweep(False)
        sleep(0.1)  # Allow time for the command to be processed
        self.write("INIT:IMM")
        wait = self.ask(":SENS:SWE:TIME?").strip()
        sleep(float(wait) + 0.1)  # Wait for the sweep to complete
        sleep(0.1)
        while int(self.ask(":STATus:OPERation:CONDition?").strip()):
            sleep(.01)
      

        # Return previous mode, incase we want to restore this
        return prev_mode

    # ------------------------------------------------------------------
    def perform_sweep(
        self,
        sweep_type: str = "LIN",
        *,
        center_frequency: float | None = None,
        frequency_span: float | None = None,
        points: int | None = None,
        if_bandwidth: float | None = None,
        segments_data: List[Dict[str, Any]] | None = None,
        freq_mode: str = "SSTOP",
        averaging: int = 1,
        measurement: str = "S21",
        format: str = "MLOG",
    ) -> tuple[np.ndarray, np.ndarray]:
        
        # backup ----------------------------------------------------------------
        old_avg_state = self.averages_enabled()
        old_avg_cnt = int(self.averages().strip())
        old_type = self.sweep_type().strip()

        try:
            # measurement selection -------------------------------------
            self.trace(1)  # select trace 1 (default)
            self.write(f"CALC:PAR1:DEF {measurement}")
            self.format(format)

            # sweep configuration --------------------------------------
            if sweep_type in {"LIN", "LOG"}:
                points = int(points)
                self.sweep_type(sweep_type)
                if center_frequency is not None:
                    self.center(center_frequency)
                if frequency_span is not None:
                    self.span(frequency_span)
                if points is not None:
                    self.points(points)
                if if_bandwidth is not None:
                    self.if_bandwidth(if_bandwidth)
            elif sweep_type == "SEGM":
                if segments_data is None:
                    raise ValueError("segments_data required for segmented sweep")
                self.sweep_type("SEGM")
                self.create_segments(segments_data)
            else:
                raise NotImplementedError(f"Sweep type {sweep_type} not supported")

            # averaging -------------------------------------------------
            self.averages_enabled(averaging > 1)
            self.averages(max(1, int(averaging)))

            # run sweep -------------------------------------------------
            self.run_sweep()

            data = self.get_complex_data()
            freq = self.frequency_axis()
            return data, freq
        finally:
            # restore previous settings --------------------------------
            self.averages_enabled(old_avg_state)
            self.averages(old_avg_cnt)
            self.sweep_type(old_type)

    # convenience wrappers identical to legacy driver ------------------
    def linear_sweep(
        self,
        center_frequency: float | None = None,
        frequency_span: float | None = None,
        points: int | None = None,
        if_bandwidth: float | None = None,
        averaging: int = 1,
        measurement: str = "S21",
        format: str = "MLOG",
    ):
        return self.perform_sweep(
            sweep_type="LIN",
            center_frequency=center_frequency,
            frequency_span=frequency_span,
            points=points,
            if_bandwidth=if_bandwidth,
            averaging=averaging,
            measurement=measurement,
            format=format,
        )

    def segmented_sweep(
        self,
        segments_data: List[Dict[str, Any]],
        *,
        freq_mode: str = "SSTOP",
        averaging: int = 1,
        measurement: str = "S21",
        format: str = "MLOG",
    ):
        return self.perform_sweep(
            sweep_type="SEGM",
            segments_data=segments_data,
            freq_mode=freq_mode,
            averaging=averaging,
            measurement=measurement,
            format=format,
        )

    # ------------------------------------------------------------------
    def create_standard_segments(
        self,
        resonator_frequencies: Sequence[float],
        *,
        fwhm_values: Sequence[float] | None = None,
        f_sec: Sequence[float] | None = None,
        n_points: Sequence[int] | None = None,
        ifbw_narrow: float = 100,
        ifbw_wide: float = 1e3,
    ) -> List[Dict[str, Union[int, float, str]]]:
        """Utility replicating MATLAB helper from original driver."""
        if fwhm_values is None:
            fwhm_values = [f * 0.01 for f in resonator_frequencies]
        if f_sec is None:
            f_sec = [0.5, 3, 30, 150]
        if n_points is None:
            n_points = [75, 60, 40, 50]
        segments: List[Dict[str, Any]] = []
        for f0, fwhm in zip(resonator_frequencies, fwhm_values):
            # ultra‑narrow around the peak
            segments.append(dict(state="ON", points=n_points[0], span=fwhm, center=f0, ifbw=ifbw_narrow))
            # wider span tiers -------------------------------------------------
            for sec, pts in zip(f_sec[1:], n_points[1:]):
                segments.append(dict(state="ON", points=pts, span=fwhm * sec, center=f0, ifbw=ifbw_wide))
        return segments



