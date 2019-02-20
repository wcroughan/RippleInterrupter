# System imports
import threading
import time
import numpy as np
import logging
from multiprocessing import Queue

# Local imports
import Logger
import Visualization
import SpikeAnalysis
import RippleAnalysis
import PositionAnalysis
import TrodesInterface

# Constant declarations
# TODO: Could all be moved to a separate file
N_POSITION_BINS = (6, 6)
# Main function that launches threads for detecting ripples, creating place
# fields and analyzing replays online.

def nTrodeAndClusterToIDMap(ntrode_id, cluster_id):
    """
    Takes in tetrode ID and cluster ID and maps it to a scalar index that can
    then be used to access the place field for the cluster.
    """

    raise NotImplementedError()

if (__name__ == "__main__"):
    # TODO: Add a config file for reading a list of tetrodes that we want to
    # work with.
    cluster_filename = "./test_clusters.trodesClusters"
    tetrodes_of_interest = [3, 14]
    n_units, cluster_identity_map = SpikeAnalysis.readClusterFile(cluster_filename, tetrodes_of_interest)
    print(cluster_identity_map)

    # TODO: Making this a giant array might not be the best idea.. Potential
    # bugs accessing it too.
    place_fields = np.zeros((n_units, N_POSITION_BINS[0], N_POSITION_BINS[1]), \
            dtype=[('nspikes', 'u4'), ('occupancy', 'f8')])

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

    # TODO: Put everything in a try/catch loop to better handle user interrupts
    # Create a buffer for spikes to be accessed until they are taken out of the
    # queue by the Bayesian Estimator.
    #spike_buffer = Queue()
    #position_buffer = Queue()

    place_field_lock = threading.Condition()
    # Initialize threads for looking at the actual/decoded position
    spike_listener      = SpikeAnalysis.SpikeDetector(sg_client, tetrode_argument, cluster_identity_map)
    position_estimator  = PositionAnalysis.PositionEstimator(sg_client, N_POSITION_BINS)
    # place_field_handler = SpikeAnalysis.PlaceFieldHandler(position_estimator, spike_listener, place_fields, \
    #         place_field_lock)
    place_field_handler = SpikeAnalysis.PlaceFieldHandler(position_estimator, spike_listener, place_fields)
    """
    bayesian_estimator  = PositionEstimator.BayesianEstimator(spike_buffer, place_fields)
    """

    # Spawn threads for handling all the place fields. We can convert this into
    # separate threads for separate fields too but that seems overkill at this
    # point.

    spike_listener.start()
    position_estimator.start()
    place_field_handler.start()
    ripple_detector.start()
    ripple_trigger.start()

    # Join all the threads to wait for their execution to  finish
    spike_listener.join()
    position_estimator.join()
    place_field_handler.join()
    ripple_detector.join()
    ripple_trigger.join()

    # For each unit detected (God knows how this will work out!), launch a
    # thread for constructing place fields.
    # TODO: This can definitely not be hardcoded like this
    n_units = 100
    place_fields = []
    for unit_id in range(n_units):
        place_fields.append(SpikeAnalysis.PlaceField(unit_id))

    print("Program finished. Exiting.")
