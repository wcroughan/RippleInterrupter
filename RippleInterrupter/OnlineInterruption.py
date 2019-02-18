# System imports
import threading
import time

# Local imports
import Logger
import SGClient
import Visualization
import SpikeAnalysis
import RippleAnalysis

# Main function that launches threads for detecting ripples, creating place
# fields and analyzing replays online.
if (__name__ == "__main__"):
    # TODO: Add a config file for reading a list of tetrodes that we want to
    # work with.
    tetrodes_of_interest = [2, 3, 14, 17, 18, 21, 22, 23]

    # Open connection to trodes.
    sg_client = SGClient("ReplayInterruption")

    # Start threads for collecting spikes and LFP
    ripple_detector = RippleAnalysis.RippleDetector(sg_client, tetrodes_of_interest)
    spike_listener  = SpikeAnalysis.SpikeDetector(sg_client, tetrodes_of_interest)

    # For each unit detected (God knows how this will work out!), launch a
    # thread for constructing place fields.
    # TODO: This can definitely not be hardcoded like this
    n_units = 100
    place_fields = []
    for unit_id in range(n_units):
        place_fields.append(SpikeAnalysis.PlaceField(unit_id))

    print("Program finished. Exiting.")
    return
