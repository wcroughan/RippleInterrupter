# System imports
import sys
import threading
import time
import ctypes
import numpy as np
import logging
import cProfile
from multiprocessing import Queue, RawArray, Condition

# PyQt imports
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication

# Local imports
import Logger
import QtHelperUtils
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
DEFAULT_RIPPLE_TRIGGERING = True

class CommandWindow(QMainWindow):

    """
    Parent window for running all the programs
    """

    def __init__(self):
        """TODO: to be defined1. """
        QMainWindow.__init__(self)
        self.setWindowTitle('Spike Processor')
        self.statusBar().showMessage('Connect to SpikeGadgets.')
        self.setupMenus()

        # TODO: None of the thread classes have any clean up at the end... TBD
        if __debug__:
            # Create code profiler
            self.code_profiler = cProfile.Profile()
            profile_prefix = "replay_disruption_profile"
            profile_filename = time.strftime(profile_prefix + "_%Y%m%d_%H%M%S.pr")

        # Tetrode info fields
        self.n_units = 0
        self.n_tetrodes = 0
        self.cluster_identity_map = dict()
        self.tetrodes_of_interest = None

        # Shared memory buffers for passing information across threads
        self.shared_raw_lfp_buffer = None
        self.shared_ripple_buffer = None
        self.shared_place_fields = None

        # Shared arrays for stimulus calibration
        self.shared_calib_plot_means = None
        self.shared_calib_plot_std_errs = None
        self.shared_calib_plot_spike_count_buffer = None

        # Trodes connection
        self.sg_client = None

        # Synchronization conditions across threads
        self.trig_condition  = Condition()
        self.show_trigger    = Condition()
        self.calib_trigger   = Condition()
        self.calib_plot_condition = Condition()

        # Initialize containers for all the thread processors
        self.calib_plot          = None
        self.lfp_listener        = None
        self.ripple_trigger      = None
        self.spike_listener      = None
        self.ripple_detector     = None
        self.position_estimator  = None
        self.place_field_handler = None
        self.graphical_interface = None

        # Launch the main graphical interface as a widget
        self.setGeometry(100, 100, 1200, 1200)

    # Functions for saving data
    def saveFields(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def saveBayesianDecoding(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    # Functions for loading data
    def loadFields(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    # Setting plot areas
    def plotBayesianEstimate(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def setupMenus(self):
        # Set up the menu bar
        menu_bar = self.menuBar()

        # File menu - Save, Load (Processed Data), Quit
        file_menu = menu_bar.addMenu('&File')

        # =============== SAVE MENU =============== 
        save_menu = file_menu.addMenu('&Save')
        save_place_fields_action = QAction('Place &Fields', self)
        save_place_fields_action.setStatusTip('Save place fields')
        save_place_fields_action.triggered.connect(self.saveFields)

        save_bayesian_decoding_action = QAction('Bayesian &Decoding', self)
        save_bayesian_decoding_action.setStatusTip('Save bayesian decoding')
        save_bayesian_decoding_action.triggered.connect(self.saveBayesianDecoding)

        save_menu.addAction(save_place_fields_action)
        save_menu.addAction(save_bayesian_decoding_action)

        # =============== LOAD MENU =============== 
        open_menu = file_menu.addMenu('&Load')
        load_clusters_action = QAction('&Clusters', self)
        load_clusters_action.setShortcut('Ctrl+C')
        load_clusters_action.triggered.connect(self.loadClusterFile)

        load_fields_action = QAction('&Fields', self)
        load_fields_action.setShortcut('Ctrl+F')
        load_fields_action.triggered.connect(self.loadFields)

        open_menu.addAction(load_fields_action)
        open_menu.addAction(load_clusters_action)

        quit_action = QAction('&Exit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.setStatusTip('Exit Program')
        quit_action.triggered.connect(qApp.quit)

        # Add actions to the file menu
        file_menu.addAction(quit_action)

        # =============== PLOT MENU =============== 
        plot_menu = menu_bar.addMenu('&Plot')
        plot_bayesian_estimate = plot_menu.addAction('&Bayesian Estimate')
        plot_bayesian_estimate.setStatusTip('Plot bayesian estimate')
        plot_bayesian_estimate.triggered.connect(self.plotBayesianEstimate)

        # =============== PREF MENU =============== 
        preferences_menu = menu_bar.addMenu('&Preferences')
        ripple_trigger_setting = preferences_menu.addAction('&Increase Speed')
        ripple_trigger_setting.setStatusTip('Enable trigerring ripples')
        ripple_trigger_setting.setChecked(DEFAULT_RIPPLE_TRIGGERING)
        ripple_trigger_setting.triggered.connect(self.setRippleTrigger)

    def setRippleTrigger(self, state):
        if state:
            self.ripple_trigger.enable()
        else:
            self.ripple_trigger.disable()

    def connect(self):
        """
        Connect to Trodes client.
        """
        try:
            self.sg_client = TrodesInterface.SGClient("ReplayInterruption")
        except Exception as err:
            QtHelperUtils.display_warning('Unable to connect to Trodes!')
            print(err)

    def loadClusterFile(self, cluster_filename=None):
        """
        Load cluster information from a cluster file.
        """
        # Uncomment to use a hardcoded file
        # cluster_filename = "./test_clusters.trodesClusters"
        # cluster_filename = "open_field_full_config20190220_172702.trodesClusters"
        cluster_config = Configuration.read_cluster_file(cluster_filename, tetrodes_of_interest)
        if cluster_config is not None:
            self.n_units = cluster_config[0]
            self.cluster_identity_map = cluster_config[1]
            if (n_units == 0):
                print(MODULE_IDENTIFIER + 'WARNING: No clusters found in the cluster file.')
            if tetrodes_of_interest is None:
                self.tetrodes_of_interest = list(self.cluster_identity_map.keys())
        else:
            print("Warning: Unable to read cluster file. Using default map.")
            self.n_units = 1
            self.cluster_identity_map = dict()
            self.cluster_identity_map[2] = {1: 0}
            self.cluster_identity_map[14] = {}
        QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Read cluster identity map.')
        print(self.cluster_identity_map)

        # Uncomment to let the user select a file
        # n_units, self.cluster_identity_map = SpikeAnalysis.readClusterFile(tetrodes=tetrodes_of_interest)
        # print(self.cluster_identity_map)

        # NOTE: Using all the tetrodes that have clusters marked on them for ripple analysis
        self.n_tetrodes = len(self.cluster_identity_map)

    def stopThreads(self):
        if __debug__:
            self.code_profiler.enable()
        try:
            # Join all the threads to wait for their execution to  finish
            # Run cleanup here
            graphical_interface.join()
            if __debug__:
                self.code_profiler.disable()
                self.code_profiler.dump_stats(profile_filename)
            logging.info(MODULE_IDENTIFIER + "GUI terminated")
            spike_listener.join()
            logging.info(MODULE_IDENTIFIER  + "Spike Listener Stopped")
            position_estimator.join()
            logging.info(MODULE_IDENTIFIER + "Position data collection Stopped")
            place_field_handler.join()
            logging.info(MODULE_IDENTIFIER + "Place field builder Stopped")
            lfp_listener.join()
            logging.info(MODULE_IDENTIFIER + "Spike calibration plot Stopped.")
            calib_plot.join()
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
            self.sg_client.closeConnections()
            print(MODULE_IDENTIFIER + "Program finished. Exiting.")
            del sg_client

    def startThreads(self):
        """
        Start all the processing thread.
        """
        tetrode_argument = [str(tet) for tet in self.tetrodes_of_interest]
        try:
            # Add conditions for which of the these threads are actually requested by the user
            self.lfp_listener = RippleAnalysis.LFPListener(self.sg_client, tetrode_argument)
            self.spike_listener = SpikeAnalysis.SpikeDetector(self.sg_client, self.cluster_identity_map)
            self.calib_plot     = CalibrationPlot.CalibrationPlot(sg_client, (shared_calib_plot_means, shared_calib_plot_std_errs, shared_calib_plot_spike_count_buffer), spike_listener, calib_plot_condition)
            self.ripple_detector = RippleAnalysis.RippleDetector(lfp_listener, calib_plot, \
                    trigger_condition=(trig_condition, show_trigger, calib_trigger), \
                    shared_buffers=(shared_raw_lfp_buffer, shared_ripple_buffer))

            # Initialize threads for looking at the actual/decoded position
            self.position_estimator  = PositionAnalysis.PositionEstimator(sg_client)
            self.place_field_handler = SpikeAnalysis.PlaceFieldHandler(position_estimator, spike_listener, shared_place_fields)
            self.ripple_trigger = RippleAnalysis.RippleSynchronizer(trig_condition, spike_listener, position_estimator, place_field_handler)
            self.graphical_interface = Visualization.GraphicsManager((shared_raw_lfp_buffer, shared_ripple_buffer), (shared_calib_plot_means, shared_calib_plot_std_errs), spike_listener, \
                    position_estimator, place_field_handler, ripple_trigger, show_trigger, calib_trigger, shared_place_fields)
        except Exception as err:
            print(err)
            return

        self.setCentralWidget(self.graphical_interface.widget)
        self.graphical_interface.start()
        self.lfp_listener.start()
        self.ripple_detector.start()
        self.spike_listener.start()
        self.position_estimator.start()
        self.place_field_handler.start()
        self.ripple_trigger.start()
        self.calib_plot.start()

        if DEFAULT_RIPPLE_TRIGGERING:
            self.ripple_trigger.enable()
        else:
            self.ripple_trigger.disable()

    def initLFPThreads(self):
        """
        Initialize the threads needed for LFP processing.
        """
        self.shared_raw_lfp_buffer = RawArray(ctypes.c_double, n_tetrodes * RiD.LFP_BUFFER_LENGTH)
        self.shared_ripple_buffer = RawArray(ctypes.c_double, n_tetrodes * RiD.RIPPLE_POWER_BUFFER_LENGTH)

        pass

    def initSpikeThread(self):
        """
        Initialize thread needed for spike processing
        """
        self.shared_place_fields = RawArray(ctypes.c_double, n_units * PositionAnalysis.N_POSITION_BINS[0] * \
                PositionAnalysis.N_POSITION_BINS[1])

    def initCalibrationThreads(self):
        """
        Initialize threads needed for stimulus calibration.
        """
        self.shared_calib_plot_means = RawArray(ctypes.c_double, RiD.CALIB_PLOT_BUFFER_LENGTH)
        self.shared_calib_plot_std_errs = RawArray(ctypes.c_double, RiD.CALIB_PLOT_BUFFER_LENGTH)
        self.shared_calib_plot_spike_count_buffer = RawArray(ctypes.c_uint32, RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE)

def main():
    # Start logging before anything else
    log_file_prefix = "replay_disruption_log"
    log_filename = time.strftime(log_file_prefix + "_%Y%m%d_%H%M%S.log")
    logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
            level=logging.INFO, datefmt="%H:%M:%S")
    # logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
    #         level=logging.DEBUG, datefmt="%H:%M:%S")
    logging.debug(MODULE_IDENTIFIER + "Starting Log file at " + time.ctime())

    qt_args = list()
    qt_args.append('-style')
    qt_args.append('Windows')
    print(MODULE_IDENTIFIER + "Qt Arguments: " + str(qt_args))
    parent_app = QApplication(qt_args)
    print(MODULE_IDENTIFIER + "Parsing Input Arguments: " + str(sys.argv))
    command_window = CommandWindow()
    command_window.show()
    sys.exit(parent_app.exec_())

if (__name__ == "__main__"):
    main()
