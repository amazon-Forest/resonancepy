import numpy as np
from qcodes.instrument import Instrument
from qcodes.validators import Arrays, Numbers, Bool, Enum, Ints
from typing import Any, Sequence, TYPE_CHECKING, Dict, List, Union
from resonance_py.utils.sim import resonator_response as rr

class SimulatedVNABase(Instrument):
    def get_idn(self) -> dict[str, str | None]:
        return {
            "vendor": "LPS",
            "model": "Dummy",
            "serial": "NA",
            "firmware": "NA",
        }

class SimulatedVNA(SimulatedVNABase):
    """
    This is the python driver for a dummy Vector Network Analyzer

    Usage:
    Initialise with
    <name> = instruments.create('<name>')

    """

    def __init__(self, name):
        """
        Initializes

        Input:
        """
        self.__name__ = __name__
        super().__init__(name)
        # self.func = get_resonance_curve
        self.max_freq = 10e9
        self.min_freq = 0

        self.sweeptime_averages = 1
        # self.span = self.stopfreq - self.startfreq

        self.fo = []
        self.Qi = []
        self.Qc_mag = []
        self.phi = []
        self.num_resonators = len(self.fo)
        self.noise_level=1e-4
        self.baseline_ripple=.01
        self.baseline_period=.1

        self.power = 0
    
        self.add_parameter(
            "sweep_type",
            label="Sweep Type",
            set_cmd=None,
            get_cmd=None,
            vals=Enum("LIN", "LOG", "SEGM", "CW"),
        )

        self.add_parameter(
            "center",
            label="Center Frequency",
            unit="Hz",
            set_cmd=None,
            get_cmd=None,
            vals=Numbers(self.min_freq, self.max_freq),
        )

        self.add_parameter(
            "span",
            label="Frequency Span",
            unit="Hz",
            set_cmd=self.set_span,
            get_cmd=None,
            vals=Numbers(0, self.max_freq),
        )

        self.add_parameter(
            "start",
            label="Start Frequency",
            unit="Hz",
            set_cmd=None,
            get_cmd=None,
            initial_value=194,
            vals=Numbers(0, self.max_freq),
        )

        self.add_parameter(
            "stop",
            label="stop Frequency",
            unit="Hz",
            set_cmd=None,
            get_cmd=None,
            vals=Numbers(0, self.max_freq),
        )

        self.add_parameter(
            "points",
            label="Sweep Points",
            set_cmd=None,
            get_cmd=None,
            vals=Ints(1, 16001),
        )

        self.add_parameter(
            "if_bandwidth",
            label="IF Bandwidth",
            unit="Hz",
            set_cmd=None,
            get_cmd=None,
            vals=Numbers(1, 3e6),
        )

        # averaging ------------------------------------------------------------------
        self.add_parameter(
            "averages_enabled",
            label="Averaging ON/OFF",
            set_cmd=None,
            get_cmd=None,
            val_mapping={True: 1, False: 0},
        )

        self.add_parameter(
            "averages",
            label="Averaging Count",
            set_cmd=None,
            get_cmd=None,
            vals=Ints(1, 65535),
        )

        # auto/hold sweep -------------------------------------------------------------
        self.add_parameter(
            "auto_sweep",
            label="Continuous Sweep",
            set_cmd=None,
            get_cmd=None,
            val_mapping={True: 1, False: 0},
        )

        # active trace / measurement --------------------------------------------------
        self.add_parameter(
            "trace",
            label="Active Measurement",
            set_cmd=None,
            get_cmd= None,
        )

        self.add_parameter(
            "format",
            label="Display Format",
            set_cmd=None,
            get_cmd=None,
            vals=Enum(
                "MLOG", "PHAS", "SMIT", "POL", "GDEL", "MLIN", "SWR", "REAL", "IMAG",
                "SLIN", "SLOG", "SCOM", "SADM", "PLIN", "PLOG",
            ),
        )

        self.add_parameter(
            "data_format",
            label="Data Output Format",
            set_cmd=None,
            get_cmd=None,
            vals=Enum(
                "ASC", "REAL", "REAL32",
            ),
        )

        # high‑level helper parameters ------------------------------------------------
        # (their *names* must match the legacy driver)
        # self.remove_parameter("frequency_axis", raise_if_missing=False)

    def frequency_axis(self):
        return np.linspace(self.start(), self.stop(), self.points())
    
    def get_s21(self,f, fr, Qi, Qc_mag, phi):
        re_inv_Qc = np.cos(phi) / Qc_mag
        inv_Ql = (1.0 / Qi) + re_inv_Qc
        Ql = 1.0 / inv_Ql
        Qc_complex = Qc_mag * np.exp(-1j * phi)
        # Calculate the S21 parameter
        S21 = 1 - (Ql / Qc_complex) / (1 + 2j * Ql * (f - fr) / fr)
        return S21
    
    def set_span(self, value):
        # self.span(value)
        center = self.center()
        startfreq = center - value / 2
        stopfreq = center + value / 2

        self.start(startfreq)
        self.stop(stopfreq)
        
        return None

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
        # self.segment_add: Parameter = self.add_parameter(
        #     "segment_add", label="Add Segment", set_cmd="SENS:SEGM:ADD")
        # self.segment_count: Parameter = self.add_parameter(
        #     "segment_count", label="Segment Count", get_cmd="SENS:SEGM:COUN?")
        # self.segment_delete: Parameter = self.add_parameter(
        #     "segment_delete", label="Delete Segment", set_cmd="SENS:SEGM:DEL {0}")
        # self.segment_list: Parameter = self.add_parameter(
        #     "segment_list", label="List Segments", get_cmd=self._get_segment_list)
    
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

        try:

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
                self.segments = segments_data
            else:
                raise NotImplementedError(f"Sweep type {sweep_type} not supported")

            def rand_num(value, min_factor=4e2, max_factor=1e4):
                return value / np.random.randint(min_factor, max_factor, dtype=np.int64)

            if self.num_resonators == 0:
                self.fo = np.random.uniform(4e9, 5e9, 3)
                self.Qi = [int(rand_num(x)) for x in self.fo]
                self.Qc_mag = [int(rand_num(x, 1e4, 2e4)) for x in self.fo]
                self.phi = [-0.45] * len(self.fo)
                self.num_resonators = len(self.fo)
            
            Qi = self.Qi
            Qc_mag = self.Qc_mag    
            phi = self.phi
            fo = self.fo
            noise_level = self.noise_level
            baseline_ripple = self.baseline_ripple
            baseline_period = self.baseline_period

            freq = self.frequency_axis()
            data = rr.generate_multi_resonator_s21(
                freq=freq,
                resonance_frequencies=fo,
                Qi_values=Qi,
                Qc_mag_values=Qc_mag,
                phi_values=phi,
                noise_level=noise_level,
                baseline_ripple=baseline_ripple,
                baseline_period=baseline_period,
            )


            return data, freq
        finally:
            # restore previous settings --------------------------------
            pass


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

