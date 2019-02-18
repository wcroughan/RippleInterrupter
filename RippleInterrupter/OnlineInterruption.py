# System imports
import threading
import time

# Local imports
import Logger
import Visualization
import SpikeAnalysis
import RippleAnalysis
import TrodesInterface

# Main function that launches threads for detecting ripples, creating place
# fields and analyzing replays online.
if (__name__ == "__main__"):
    # TODO: Add a config file for reading a list of tetrodes that we want to
    # work with.
    tetrodes_of_interest = [2, 3, 14, 17, 18, 21, 22, 23]
    # Trodes needs strings!
    tetrode_argument = [str(tet) for tet in tetrodes_of_interest]

    # Open connection to trodes.
    sg_client = TrodesInterface.SGClient("ReplayInterruption")

    # Start a thread for triggering analysis when ripple is triggered.
    trig_condition  = threading.Condition()
    ripple_trigger  = RippleAnalysis.RippleSynchronizer(trig_condition)

    # Start threads for collecting spikes and LFP
    ripple_detector = RippleAnalysis.RippleDetector(sg_client, tetrode_argument, \
            baseline_stats=[60.0, 30.0], trigger_condition=trig_condition)
    # TODO: Uncomment to start looking at spikes
    # spike_listener  = SpikeAnalysis.SpikeDetector(sg_client, tetrode_argument)

    # For each unit detected (God knows how this will work out!), launch a
    # thread for constructing place fields.
    # TODO: This can definitely not be hardcoded like this
    n_units = 100
    place_fields = []
    for unit_id in range(n_units):
        place_fields.append(SpikeAnalysis.PlaceField(unit_id))

    print("Program finished. Exiting.")
