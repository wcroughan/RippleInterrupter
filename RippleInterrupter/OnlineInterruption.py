# System imports
import sys
import threading
import time
import ctypes
import numpy as np
import logging
import cProfile
from multiprocessing import Queue, RawArray, Condition

# Local imports
import Logger
import Configuration
import Visualization
import SpikeAnalysis
import RippleAnalysis
import PositionAnalysis
import TrodesInterface
import PositionDecoding
import RippleDefinitions as RiD
import CalibrationPlot

MODULE_IDENTIFIER = "[OnlineInterruption] "

def main():
    # TODO: None of the thread classes have any clean up at the end... TBD
    # Start logging before anything else
    log_file_prefix = "replay_disruption_log"
    log_filename = time.strftime(log_file_prefix + "_%Y%m%d_%H%M%S.log")
    logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
            level=logging.INFO, datefmt="%H:%M:%S")
    logging.debug(MODULE_IDENTIFIER + "Starting Log file at " + time.ctime())

    if __debug__:
        # Create code profiler
        code_profiler = cProfile.Profile()
        profile_prefix = "replay_disruption_profile"
        profile_filename = time.strftime(profile_prefix + "_%Y%m%d_%H%M%S.pr")

    # Not necessary to add a filename here. Can be read using a dialog box now
    tetrodes_of_interest = [2, 14]

    # Uncomment to use a hardcoded file
    # cluster_filename = "./test_clusters.trodesClusters"
    # cluster_filename = "open_field_full_config20190220_172702.trodesClusters"
    cluster_filename = None
    cluster_config = Configuration.readClusterFile(cluster_filename, tetrodes_of_interest)
    if cluster_config is not None:
        n_units = cluster_config[0]
        cluster_identity_map = cluster_config[1]
    else:
        print("Warning: Unable to read cluster file. Using default map.")
        n_units = 1
        cluster_identity_map = dict()
        cluster_identity_map[2] = {1: 0}
        cluster_identity_map[14] = {}
    print(cluster_identity_map)

    # Uncomment to let the user select a file
    # n_units, cluster_identity_map = SpikeAnalysis.readClusterFile(tetrodes=tetrodes_of_interest)
    # print(cluster_identity_map)

    # NOTE: Using all the tetrodes that have clusters marked on them for ripple analysis
    n_tetrodes = len(cluster_identity_map)
    shared_raw_lfp_buffer = RawArray(ctypes.c_double, n_tetrodes * RiD.LFP_BUFFER_LENGTH)
    shared_ripple_buffer = RawArray(ctypes.c_double, n_tetrodes * RiD.RIPPLE_POWER_BUFFER_LENGTH)
    shared_place_fields = RawArray(ctypes.c_double, n_units * PositionAnalysis.N_POSITION_BINS[0] * \
            PositionAnalysis.N_POSITION_BINS[1])
    shared_calib_plot_means = RawArray(ctypes.c_double, RiD.CALIB_PLOT_BUFFER_LENGTH)
    shared_calib_plot_std_errs = RawArray(ctypes.c_double, RiD.CALIB_PLOT_BUFFER_LENGTH)


    # Open connection to trodes.
    sg_client = TrodesInterface.SGClient("ReplayInterruption")

    # Start a thread for triggering analysis when ripple is triggered. Use a
    # separate condition to SHOW the detected ripple so that we can space the
    # visualization out from the actual detection/analysis 
    trig_condition  = Condition()
    show_trigger    = Condition()
    calib_trigger   = Condition()

    # Start threads for collecting spikes and LFP
    # Trodes needs strings!
    tetrode_argument = [str(tet) for tet in tetrodes_of_interest]
    try:
        lfp_listener = RippleAnalysis.LFPListener(sg_client, tetrode_argument)
        spike_listener      = SpikeAnalysis.SpikeDetector(sg_client, cluster_identity_map)
        calib_plot          = CalibrationPlot.CalibrationPlot(sg_client,  (shared_calib_plot_means, shared_calib_plot_std_errs), spike_listener)

        ripple_detector = RippleAnalysis.RippleDetector(lfp_listener, calib_plot, \
                trigger_condition=(trig_condition, show_trigger, calib_trigger), \
                shared_buffers=(shared_raw_lfp_buffer, shared_ripple_buffer))

        # Initialize threads for looking at the actual/decoded position
        position_estimator  = PositionAnalysis.PositionEstimator(sg_client)
        place_field_handler = SpikeAnalysis.PlaceFieldHandler(position_estimator, spike_listener, shared_place_fields)
        ripple_trigger      = RippleAnalysis.RippleSynchronizer(trig_condition, spike_listener, position_estimator, place_field_handler)
    except Exception as err:
        print(err)
        return

    # Optionally, launch a graphics thread for continuously monitoring
    # different threads for spikes, position data and ripples and show them to
    # the user in real time.
    graphical_interface = Visualization.GraphicsManager((shared_raw_lfp_buffer, shared_ripple_buffer), (shared_calib_plot_means, shared_calib_plot_std_errs), spike_listener, \
            position_estimator, place_field_handler, ripple_trigger, show_trigger, calib_trigger, shared_place_fields)

    # Spawn threads for handling all the place fields. We can convert this into
    # separate threads for separate fields too but that seems overkill at this
    # point.

    graphical_interface.start()
    # Start code profiler... Be sure to comment this out when not profiling the code
    lfp_listener.start()
    ripple_detector.start()
    spike_listener.start()
    position_estimator.start()
    place_field_handler.start()
    ripple_trigger.start()

    # By default, enable the ripple trigerring 
    ripple_trigger.enable()

    if __debug__:
        code_profiler.enable()
    try:
        # Join all the threads to wait for their execution to  finish
        # Run cleanup here
        graphical_interface.join()
        if __debug__:
            code_profiler.disable()
            code_profiler.dump_stats(profile_filename)
        logging.info(MODULE_IDENTIFIER + "GUI terminated")
        spike_listener.join()
        logging.info(MODULE_IDENTIFIER  + "Spike Listener Stopped")
        position_estimator.join()
        logging.info(MODULE_IDENTIFIER + "Position data collection Stopped")
        place_field_handler.join()
        logging.info(MODULE_IDENTIFIER + "Place field builder Stopped")
        lfp_listener.join()
        logging.info(MODULE_IDENTIFIER + "LFP listener Stopped")
        ripple_detector.join()
        logging.info(MODULE_IDENTIFIER + "Ripple detector Stopped")
        ripple_trigger.join()
        logging.info(MODULE_IDENTIFIER + "Ripple event synchronizer Stopped")
    except (KeyboardInterrupt, SystemExit):
        logging.debug(MODULE_IDENTIFIER + "Caught Keyboard Interrupt from user...")
    finally:
        # TODO: Delete all the threads
        del shared_place_fields
        del shared_ripple_buffer
        del shared_raw_lfp_buffer
        del lfp_listener
        del spike_listener
        del position_estimator
        del place_field_handler
        del ripple_trigger
        del graphical_interface
        sg_client.closeConnections()
        print(MODULE_IDENTIFIER + "Program finished. Exiting.")
        del sg_client

if (__name__ == "__main__"):
    main()
