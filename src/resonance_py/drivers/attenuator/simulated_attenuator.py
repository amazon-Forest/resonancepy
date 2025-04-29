import numpy as np
from qcodes.instrument import Instrument
from qcodes.validators import Arrays, Numbers, Bool, Enum, Ints
from typing import Any, Sequence, TYPE_CHECKING, Dict, List, Union
from resonance_py.utils.sim import resonator_response as rr


class SimulatedAttenuatorBase(Instrument):
    def get_idn(self) -> dict[str, str | None]:
        return {
            "vendor": "LPS",
            "model": "Dummy",
            "serial": "NA",
            "firmware": "NA",
        }
    

class SimulatedAttenuator(SimulatedAttenuatorBase):
    """
    Simulated attenuator class for testing purposes.
    This class simulates the behavior of an attenuator by generating
    a response based on the input parameters.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

        self.minimum_attenuation = 0.0
        self.maximum_attenuation = 100.0

        self.add_parameter(
            "attenuation",
            label="Attenuation (dB)",
            unit="dB",
            initial_value=0.0,
            vals=Numbers(
                self.minimum_attenuation, self.maximum_attenuation
            ),
            get_cmd=None,
            set_cmd=None,
        )


# atten = SimulatedAttenuator("attenuator")
# atten.attenuation(11)

# print(atten.attenuation())