# System imports
import sys
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
import PositionDecoding

MODULE_IDENTIFIER = "[OnlineInterruption] "

def main():
    # TODO: None of the thread classes have any clean up at the end... TBD
    # Start logging before anything else
    log_file_prefix = "replay_disruption_log"
    # self._filename = os.getcwd() + "/" + time.strftime(file_prefix + "_%Y%m%d_%H%M%S.log")
    log_filename = time.strftime(log_file_prefix + "_%Y%m%d_%H%M%S.log")
    logging.basicConfig(filename=log_filename, format="%(asctime)s:%(message)s", \
            level=logging.DEBUG, datefmt="%H:%M:%S")
    logging.debug(MODULE_IDENTIFIER + "Starting Log file at " + time.ctime())

    # Not necessary to add a filename here. Can be read using a dialog box now
    tetrodes_of_interest = [3, 14]

    # Uncomment to use a hardcoded file
    cluster_filename = "./test_clusters.trodesClusters"
    n_units, cluster_identity_map = SpikeAnalysis.readClusterFile(cluster_filename, tetrodes_of_interest)

    # Uncomment to let the user select a file
    # n_units, cluster_identity_map = SpikeAnalysis.readClusterFile(tetrodes=tetrodes_of_interest)
    # print(cluster_identity_map)

    place_fields = np.zeros((n_units, PositionAnalysis.N_POSITION_BINS[0], \
            PositionAnalysis.N_POSITION_BINS[1]), dtype='float')

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

    # Create a buffer for spikes to be accessed until they are taken out of the
    # queue by the Bayesian Estimator.
    #spike_buffer = Queue()
    #position_buffer = Queue()

    # Initialize threads for looking at the actual/decoded position
    spike_listener      = SpikeAnalysis.SpikeDetector(sg_client, cluster_identity_map)
    position_estimator  = PositionAnalysis.PositionEstimator(sg_client)
    place_field_handler = SpikeAnalysis.PlaceFieldHandler(position_estimator, spike_listener, place_fields)
    bayesian_estimator  = PositionDecoding.BayesianEstimator(spike_listener, place_fields)

    # Optionally, launch a graphics thread for continuously monitoring
    # different threads for spikes, position data and ripples and show them to
    # the user in real time.
    graphical_interface = Visualization.GraphicsManager(ripple_detector, spike_listener, position_estimator, \
            place_field_handler, trig_condition)

    # Spawn threads for handling all the place fields. We can convert this into
    # separate threads for separate fields too but that seems overkill at this
    # point.

    graphical_interface.start()
    spike_listener.start()
    position_estimator.start()
    place_field_handler.start()
    """
    ripple_detector.start()
    ripple_trigger.start()
    """

    try:
        # Join all the threads to wait for their execution to  finish
        # Run cleanup here
        # graphical_interface.terminate()
        graphical_interface.join()
        logging.debug(MODULE_IDENTIFIER + "GUI terminated")
        spike_listener.join()
        logging.debug(MODULE_IDENTIFIER  + "Spike Listener Stopped")
        position_estimator.join()
        logging.debug(MODULE_IDENTIFIER + "Position data collection Stopped")
        place_field_handler.join()
        logging.debug(MODULE_IDENTIFIER + "Place field builder Stopped")
        """
        ripple_detector.join()
        ripple_trigger.join()
        """
    except (KeyboardInterrupt, SystemExit):
        logging.debug(MODULE_IDENTIFIER + "Caught Keyboard Interrupt from user...")
    finally:
        # TODO: Delete all the threads
        logging.debug(MODULE_IDENTIFIER + "Program finished. Exiting.")


if (__name__ == "__main__"):
    main()
