import pyvisa
import time

def set_attenuation(target_attenuation_value: int, address:str):
    """
    Sets the attenuation on the Attenuation Control Unit (ACU) to the specified
    target_attenuation_value and verifies that it has been set properly.

    Parameters
    ----------
    target_attenuation_value : float
        Desired attenuation value to set on the ACU.

    Returns
    -------
    float
        The current attenuation value that the ACU reports.
    """
    # Create a PyVISA resource manager
    rm = pyvisa.ResourceManager()

    # Define the VISA address (same as the MATLAB visaString: 'GPIB0::28::0::INSTR')
    #visa_address = 'TCPIP0::k-j7201c-400103.local::inst0::INSTR'

    # Open a connection to the instrument
    ACU = rm.open_resource(address)

    # Set a 30-second timeout (in milliseconds for PyVISA)
    ACU.timeout = 30000

    # Set the attenuation value
    ACU.write(f"ATT {target_attenuation_value}")
    time.sleep(0.5)

    # Attempt to confirm the attenuation value up to max_attempts times
    max_attempts = 10
    curr_atten_value = None

    for attempt_num in range(1, max_attempts + 1):
        # Query the current attenuation value
        response = ACU.query("ATT?")
        try:
            curr_atten_value = float(response)
        except ValueError:
            # If the response isn't a number, just set curr_atten_value to None
            curr_atten_value = None

        # If the current attenuation matches the target, break out of the loop
        if curr_atten_value == target_attenuation_value:
            break
        else:
            time.sleep(0.5)
            # Send the command again
            ACU.write(f"ATT {target_attenuation_value}")
            time.sleep(0.5)

    # If we reach the maximum number of attempts, raise an error
    if curr_atten_value != target_attenuation_value:
        ACU.close()
        raise ValueError(
            f"ERROR: target ({target_attenuation_value}) and current ({curr_atten_value}) "
            "attenuation values do not match."
        )

    # Close the instrument connection
    ACU.close()

    # Return the (verified) current attenuation value
    return curr_atten_value

