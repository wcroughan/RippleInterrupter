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
from PyQt5 import QtCore

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
DEFAULT_SERIAL_ENABLED = False
DEFAULT_STIM_MODE_MANUAL_ENABLED = True
DEFAULT_STIM_MODE_POSITION_ENABLED = False
DEFAULT_STIM_MODE_RIPPLE_ENABLED = False

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
            self.profile_filename = time.strftime(profile_prefix + "_%Y%m%d_%H%M%S.pr")

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
        self.data_streaming = False

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
        self.setGeometry(100, 100, 1000, 800)

        # enable custom window hint
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)

        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)

    # Functions for saving data
    def saveFields(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def saveBayesianDecoding(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    # Functions for loading data
    def loadFields(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    # Set up the serial port
    def enableSerialPort(self):
        QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Enabling serial port.')
        # TODO: To the information statement above, add a line telling which
        # port is currently being used.

        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def testSingleSerialPulse(self):
        # TODO: Send a single pulse on the serial port to test port functionality.
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def testContinuousSerialPulse(self):
        # TODO: Send a series of pulses on the serial port to monitor the
        # hardware and test if it is working.
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    # Setting plot areas
    def plotBayesianEstimate(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def disconnectAndQuit(self):
        if self.graphical_interface is not None:
            self.graphical_interface.kill_gui()
        self.stopThreads()
        qApp.quit()

    def setupMenus(self):
        # Set up the menu bar
        menu_bar = self.menuBar()

        # File menu - Save, Load (Processed Data), Quit
        file_menu = menu_bar.addMenu('&File')

        connect_action = file_menu.addAction('&Connect')
        connect_action.setShortcut('Ctrl+N')
        connect_action.setStatusTip('Connect SpikeGadgets')
        connect_action.triggered.connect(self.connectSpikeGadgets)

        stream_action = file_menu.addAction('S&tream')
        stream_action.setShortcut('Ctrl+T')
        stream_action.setStatusTip('Stream data')
        stream_action.triggered.connect(self.streamData)

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
        quit_action.triggered.connect(self.disconnectAndQuit)

        # Add actions to the file menu
        file_menu.addAction(quit_action)

        # =============== PLOT MENU =============== 
        plot_menu = menu_bar.addMenu('&Plot')
        plot_bayesian_estimate = plot_menu.addAction('&Bayesian Estimate')
        plot_bayesian_estimate.setStatusTip('Plot bayesian estimate')
        plot_bayesian_estimate.triggered.connect(self.plotBayesianEstimate)

        # =============== SERIAL MENU =============== 
        serial_menu = menu_bar.addMenu('&Serial')
        enable_serial_port = serial_menu.addAction('&Enable Port')
        enable_serial_port.setStatusTip('Enable default serial port.')
        enable_serial_port.triggered.connect(self.enableSerialPort)
        enable_serial_port.setChecked(DEFAULT_SERIAL_ENABLED)

        test_single_pulse = serial_menu.addAction('Test &Single')
        test_single_pulse.setStatusTip('Send single biphasic pulse on serial port.')
        test_single_pulse.triggered.connect(self.testSingleSerialPulse)

        test_continuous_pulse = serial_menu.addAction('Test &Continuous')
        test_continuous_pulse.setStatusTip('Send a stream of biphasic pulses on the serial port.')
        test_continuous_pulse.triggered.connect(self.testContinuousSerialPulse)

        # =============== STIM MENU =============== 
        stimulation_menu = menu_bar.addMenu('&Stimulation')
        stim_mode_manual = stimulation_menu.addAction('&Manual')
        stim_mode_manual.setStatusTip('Set stimulation mode to manual.')
        stim_mode_manual.setChecked(DEFAULT_STIM_MODE_MANUAL_ENABLED)

        stim_mode_position = stimulation_menu.addAction('&Position')
        stim_mode_position.setStatusTip('Use position and velocity to simulate.')
        stim_mode_position.setChecked(DEFAULT_STIM_MODE_POSITION_ENABLED)

        stim_mode_ripple = stimulation_menu.addAction('&Ripple')
        stim_mode_ripple.setStatusTip('Stimulate on Sharp-Wave Ripples.')
        stim_mode_ripple.setChecked(DEFAULT_STIM_MODE_RIPPLE_ENABLED)

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

    def connectSpikeGadgets(self):
        """
        Connect to Trodes client.
        """
        try:
            self.sg_client = TrodesInterface.SGClient("ReplayInterruption")
        except Exception as err:
            QtHelperUtils.display_warning('Unable to connect to Trodes!')
            print(err)
            return
        if not self.cluster_identity_map:
            self.loadClusterFile()

        try:
            # TODO: Use preferences to selectively start the desired threads
            # LFP Threads
            self.initLFPThreads()

            # Position data
            self.position_estimator  = PositionAnalysis.PositionEstimator(self.sg_client)

            # Spike data
            self.initSpikeThreads()

            # Calibration data
            self.initCalibrationThreads()

            # Ripple triggered actions
            self.initRippleTriggerThreads()

            self.graphical_interface = Visualization.GraphicsManager((self.shared_raw_lfp_buffer,\
                    self.shared_ripple_buffer), (self.shared_calib_plot_means, self.shared_calib_plot_std_errs),\
                    self.spike_listener, self.position_estimator, self.place_field_handler, self.ripple_trigger,\
                    self.show_trigger, self.calib_trigger, self.shared_place_fields)
        except Exception as err:
            print(err)
            return
        self.setCentralWidget(self.graphical_interface.widget)
        self.statusBar().showMessage('Connected to SpikeGadgets. Press Ctrl+T to stream.')

    def loadClusterFile(self, cluster_filename=None):
        """
        Load cluster information from a cluster file.
        """
        # Uncomment to use a hardcoded file
        cluster_filename = "./config/full_config20190304.trodesClusters"
        # cluster_filename = "open_field_full_config20190220_172702.trodesClusters"
        cluster_config = Configuration.read_cluster_file(cluster_filename, self.tetrodes_of_interest)
        if cluster_config is not None:
            self.n_units = cluster_config[0]
            self.cluster_identity_map = cluster_config[1]
            if (self.n_units == 0):
                print(MODULE_IDENTIFIER + 'WARNING: No clusters found in the cluster file.')
            if self.tetrodes_of_interest is None:
                self.tetrodes_of_interest = list(self.cluster_identity_map.keys())
        else:
            print("Warning: Unable to read cluster file. Using default map.")
            self.n_units = 1
            self.cluster_identity_map = dict()
            self.cluster_identity_map[2] = {1: 0}
            self.cluster_identity_map[14] = {}

        if __debug__:
            QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Read cluster identity map.')
        print(self.cluster_identity_map)

        # Uncomment to let the user select a file
        # n_units, self.cluster_identity_map = SpikeAnalysis.readClusterFile(tetrodes=tetrodes_of_interest)
        # print(self.cluster_identity_map)

        # NOTE: Using all the tetrodes that have clusters marked on them for ripple analysis
        self.n_tetrodes = len(self.cluster_identity_map)

    def streamData(self):
        if self.data_streaming:
            self.stopThreads()
            self.data_streaming = False
        else:
            self.startThreads()
            self.data_streaming = True

    def stopThreads(self):
        try:
            # Join all the threads to wait for their execution to  finish
            # Run cleanup here
            self.graphical_interface.join()
            if __debug__:
                self.code_profiler.disable()
                self.code_profiler.dump_stats(self.profile_filename)
            logging.info(MODULE_IDENTIFIER + "GUI terminated")
            self.spike_listener.join()
            logging.info(MODULE_IDENTIFIER  + "Spike Listener Stopped")
            self.position_estimator.join()
            logging.info(MODULE_IDENTIFIER + "Position data collection Stopped")
            self.place_field_handler.join()
            logging.info(MODULE_IDENTIFIER + "Place field builder Stopped")
            self.lfp_listener.join()
            logging.info(MODULE_IDENTIFIER + "Spike calibration plot Stopped.")
            self.calib_plot.join()
            logging.info(MODULE_IDENTIFIER + "LFP listener Stopped")
            self.ripple_detector.join()
            logging.info(MODULE_IDENTIFIER + "Ripple detector Stopped")
            self.ripple_trigger.join()
            logging.info(MODULE_IDENTIFIER + "Ripple event synchronizer Stopped")
            self.sg_client.closeConnections()
        except Exception as err:
            logging.debug(MODULE_IDENTIFIER + "Caught Interrupt while exiting...")
            print(err)
        print(MODULE_IDENTIFIER + "Program finished. Exiting.")

    def startThreads(self):
        """
        Start all the processing thread.
        """
        try:
            self.graphical_interface.start()
            self.position_estimator.start()
            self.lfp_listener.start()
            self.ripple_detector.start()
            self.spike_listener.start()
            self.place_field_handler.start()
            self.ripple_trigger.start()
            self.calib_plot.start()
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to stream. Check connection to client!')
            print(err)
            return

        if DEFAULT_RIPPLE_TRIGGERING:
            self.ripple_trigger.enable()
        else:
            self.ripple_trigger.disable()
        self.statusBar().showMessage('Streaming!')
        if __debug__:
            self.code_profiler.enable()

    def initRippleTriggerThreads(self):
        """
        Initialize threads dependent on ripple triggers (these need pretty much everything!)
        """
        try:
            self.ripple_detector = RippleAnalysis.RippleDetector(self.lfp_listener, self.calib_plot,\
                    trigger_condition=(self.trig_condition, self.show_trigger, self.calib_trigger),\
                    shared_buffers=(self.shared_raw_lfp_buffer, self.shared_ripple_buffer))
            self.ripple_trigger = RippleAnalysis.RippleSynchronizer(self.trig_condition, self.spike_listener,\
                    self.position_estimator, self.place_field_handler)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start ripple trigger threads(s).')
            print(err)
            return

    def initLFPThreads(self):
        """
        Initialize the threads needed for LFP processing.
        """
        self.shared_raw_lfp_buffer = RawArray(ctypes.c_double, self.n_tetrodes * RiD.LFP_BUFFER_LENGTH)
        self.shared_ripple_buffer = RawArray(ctypes.c_double, self.n_tetrodes * RiD.RIPPLE_POWER_BUFFER_LENGTH)
        try:
            tetrode_argument = [str(tet) for tet in self.tetrodes_of_interest]
            self.lfp_listener = RippleAnalysis.LFPListener(self.sg_client, tetrode_argument)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start LFP thread(s).')
            print(err)
            return

    def initSpikeThreads(self):
        """
        Initialize thread needed for spike processing
        """
        self.shared_place_fields = RawArray(ctypes.c_double, self.n_units * PositionAnalysis.N_POSITION_BINS[0] * \
                PositionAnalysis.N_POSITION_BINS[1])
        try:
            self.spike_listener = SpikeAnalysis.SpikeDetector(self.sg_client, self.cluster_identity_map)
            self.place_field_handler = SpikeAnalysis.PlaceFieldHandler(self.position_estimator,\
                    self.spike_listener, self.shared_place_fields)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start spike thread(s).')
            print(err)
            return

    def initCalibrationThreads(self):
        """
        Initialize threads needed for stimulus calibration.
        """
        self.shared_calib_plot_means = RawArray(ctypes.c_double, RiD.CALIB_PLOT_BUFFER_LENGTH)
        self.shared_calib_plot_std_errs = RawArray(ctypes.c_double, RiD.CALIB_PLOT_BUFFER_LENGTH)
        self.shared_calib_plot_spike_count_buffer = RawArray(ctypes.c_uint32, RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE)
        try:
            self.calib_plot = CalibrationPlot.CalibrationPlot(self.sg_client, (self.shared_calib_plot_means,\
                    self.shared_calib_plot_std_errs, self.shared_calib_plot_spike_count_buffer),\
                    self.spike_listener, self.calib_plot_condition)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start calibration thread.')
            print(err)
            return
def main():
    # Start logging before anything else
    log_file_prefix = "replay_disruption_log"
    log_filename = time.strftime(log_file_prefix + "_%Y%m%d_%H%M%S.log")
    if __debug__:
        logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
                level=logging.DEBUG, datefmt="%H:%M:%S")
    else:
        logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
                level=logging.INFO, datefmt="%H:%M:%S")
    logging.debug(MODULE_IDENTIFIER + "Starting Log file at " + time.ctime())

    qt_args = list()
    qt_args.append('OnlineInterruption.py')
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
