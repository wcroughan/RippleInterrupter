# System imports
import sys
import threading
import time
import ctypes
import numpy as np
import logging
import cProfile
import collections
from multiprocessing import Queue, RawArray, Condition
from multiprocessing import Pipe, Lock, Event, Value

# PyQt imports
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QMessageBox
from PyQt5 import QtCore

# Local imports
import Logger
import SerialPort
import QtHelperUtils
import Configuration
import Visualization
import SpikeAnalysis
import ThreadExtension
import RippleAnalysis
import PositionAnalysis
import TrodesInterface
import PositionDecoding
import RippleDefinitions as RiD
import CalibrationPlot

MODULE_IDENTIFIER = "[OnlineInterruption] "
# User selection choices for what they want to see on the screen
DEFAULT_LFP_CHOICE      = True
DEFAULT_SPIKES_CHOICE   = True
DEFAULT_POSITION_CHOICE = True
DEFAULT_FIELD_CHOICE    = True
DEFAULT_STIMULATION_CHOICE = False

# Choices in functionality
DEFAULT_SERIAL_ENABLED = False
DEFAULT_STIM_MODE_MANUAL_ENABLED = False
DEFAULT_STIM_MODE_POSITION_ENABLED = False
DEFAULT_STIM_MODE_RIPPLE_ENABLED = True

class StimulationSynchronizer(ThreadExtension.StoppableProcess):
    """
    Waits for a stimulation events to be detected and processes downstream changes for
    analyzing spike contents.
    """

    # Wait for 10ms while checking if the event flag is set.
    _EVENT_TIMEOUT = 1.0
    _SPIKE_BUFFER_SIZE = 200
    CLASS_IDENTIFIER = "[StimulationSynchronizer] "

    def __init__(self, sync_event, spike_listener, position_estimator, place_field_handler, sg_client=None):
        """TODO: to be defined1. """
        ThreadExtension.StoppableProcess.__init__(self)
        self._sync_event = sync_event
        self._spike_buffer = collections.deque(maxlen=self._SPIKE_BUFFER_SIZE)
        self._spike_histogram = collections.Counter()
        self._spike_buffer_connection = spike_listener.get_spike_buffer_connection()
        self._position_buffer_connection = position_estimator.get_position_buffer_connection()
        self._place_field_handler = place_field_handler
        self._sg_client = sg_client
        self._position_access = Lock()
        self._spike_access = Lock()
        # TODO: This functionality should be moved to the parent class
        self._enable_synchrnoizer = Lock()
        self._clusters_of_interest = [Configuration.EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_A[:], \
                Configuration.EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_B[:]]
        print(self._clusters_of_interest)
        self._is_disabled = Value("b", True)

        # Position data at the time ripple is triggered
        self._pos_x = -1
        self._pos_y = -1
        self._most_recent_speed = 0
        self._most_recent_pos_timestamp = 0
        self._serial_port = None
        try:
            self._serial_port = SerialPort.BiphasicPort()
        except Exception as err:
            logging.warning(self.CLASS_IDENTIFIER + "Unable to open Serial port.")
            print(err)
        logging.info(self.CLASS_IDENTIFIER + "Started Stimulation Synchronization thread.")

    def enableSerial(self):
        if self._serial_port is not None:
            self._serial_port.enable()

    def disableSerial(self):
        if self._serial_port is not None:
            self._serial_port.disable()

    def startManualStimulation(self):
        """
        Stimulate irrespective of the recording events and conditions for
        duration and period specified by the configuration.
        """
        # First make sure that the serial port is well defined and enabled
        if self._serial_port is not None:
            if self._serial_port.getStatus():
                stim_start_time = time.time()
                current_time = time.time()
                while (current_time - stim_start_time < Config.MANUAL_STIM_DURATION):
                    self._serial_port.sendBiphasicPulse()
                    time.sleep(Config.MANUAL_STIM_INTER_PULSE_INTERVAL)
                    current_time = time.time()
                    # TODO: Add the last spike/trodes timestamp to this data.
                    logging.info(CLASS_IDENTIFIER + "Delivered STIM at %.2f"%current_time)
            else:
                QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Port disbled!')
        else:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Port undefined!')

    def enable(self):
        self._enable_synchrnoizer.acquire()
        self._is_disabled.value = False
        logging.info(self.CLASS_IDENTIFIER + "Ripple disruption ENABLED.")
        self._enable_synchrnoizer.release()

    def disable(self):
        self._enable_synchrnoizer.acquire()
        self._is_disabled.value = True
        logging.info(self.CLASS_IDENTIFIER + "Ripple disruption DISABLED.")
        self._enable_synchrnoizer.release()

    def fetch_current_velocity(self):
        """
        Get recent velocity (and position) and use that to determine if the
        animal was running when the current ripple was detected.
        """
        while not self.req_stop():
            if self._position_buffer_connection.poll():
                position_data = self._position_buffer_connection.recv()
                with self._position_access:
                    self._most_recent_pos_timestamp = position_data[0]
                    self._most_recent_speed = position_data[3]
                    self._pos_x = position_data[1]
                    self._pos_y = position_data[2]
            else:
                time.sleep(0.005)

    def fetch_most_recent_spike(self):
        """
        Get the most recent spike and put it in the rotating spike buffer
        (keeps track of the last self._SPIKE_BUFFER_SIZE spikes.)
        """
        while not self.req_stop():
            if self._spike_buffer_connection.poll():
                # NOTE: spike_data received here is a tuple (cluster identity, trodes timestamp)
                spike_data = self._spike_buffer_connection.recv()
                with self._spike_access:
                    if len(self._spike_buffer) == self._SPIKE_BUFFER_SIZE:
                        removed_spike = self._spike_buffer.popleft()
                        self._spike_histogram[removed_spike[0]] -= 1
                    spike_cluster = spike_data[0]
                    if (spike_cluster in self._clusters_of_interest[0]) or \
                            (spike_cluster in self._clusters_of_interest[1]):
                        # NOTE: If this starts taking too long, can switch to default dictionary
                        self._spike_buffer.append(spike_data)
                        self._spike_histogram[spike_cluster] += 1
            else:
                # NOTE: Making the thread sleep for 5ms might not hurt but we
                # will have to find out.
                time.sleep(0.005)

    def run(self):
        # Create a thread that fetches and keeps track of the last few spikes.
        spike_fetcher = threading.Thread(name="SpikeFetcher", daemon=True, \
                target=self.fetch_most_recent_spike)
        velocity_fetcher = threading.Thread(name="VelocityFetcher", daemon=True, \
                target=self.fetch_current_velocity)

        spike_fetcher.start()
        velocity_fetcher.start()
        while not self.req_stop():
            # Check if the process has been enabled
            self._enable_synchrnoizer.acquire()
            current_state = self._is_disabled.value
            self._enable_synchrnoizer.release()
            if current_state:
                logging.debug(self.CLASS_IDENTIFIER + "Process sleeping")
                time.sleep(0.1)
                continue

            logging.debug(self.CLASS_IDENTIFIER + "Waiting for ripple trigger...")
            with self._sync_event:
                thread_notified = self._sync_event.wait(self._EVENT_TIMEOUT)

            if thread_notified:
                # TODO: Only include spikes that occurred a specific amount of time before now.
                with self._position_access:
                    # If the animal is running faster than our speed threshold, ignore the ripple
                    if self._most_recent_speed < RiD.MOVE_VELOCITY_THRESOLD:
                        self._serial_port.sendBiphasicPulse()
                        print(self.CLASS_IDENTIFIER + "Ripple tiggered. Loc (%d, %d), V %.2fcm/s" \
                                %(self._pos_x, self._pos_y, self._most_recent_speed))
                        # trodes_timestamp = self._sg_client.latestTrodesTimestamp()
                        logging.info(self.CLASS_IDENTIFIER + "Ripple tiggered. Loc (%d, %d), V %.2fcm/s. Time: %.6f" \
                                %(self._pos_x, self._pos_y, self._most_recent_speed, time.perf_counter()))

                    with self._spike_access:
                        if len(self._spike_buffer) > 0:
                            # By default, returns 10 most frequent entries
                            most_spiking_unit = self._spike_histogram.most_common()[0][0]
                            most_recent_spike_time = self._spike_buffer[-1][1]
                            logging.info(self.CLASS_IDENTIFIER + "Most recent spike at %d"%most_recent_spike_time)
                            print(self._spike_histogram)
                        else:
                            logging.debug(self.CLASS_IDENTIFIER + "Spike buffer empty!")

                # DEBUGGING: Print spike count from each of the clusters
                # print(self._place_field_handler.get_peak_firing_location(most_spiking_unit))

        logging.info(self.CLASS_IDENTIFIER + "Main process exited.")
        spike_fetcher.join()
        velocity_fetcher.join()
        logging.info(self.CLASS_IDENTIFIER + "Helper threads exited.")
        logging.info(MODULE_IDENTIFIER + "Stimulation event synchronizer Stopped")

class CommandWindow(QMainWindow):
    """
    Parent window for running all the programs
    """

    def __init__(self):
        """TODO: to be defined1. """
        QMainWindow.__init__(self)
        self.setWindowTitle('Spike Processor')
        self.statusBar().showMessage('Connect to SpikeGadgets.')
        self.stim_mode_position = None
        self.stim_mode_ripple = None
        self.setupMenus()

        # TODO: None of the thread classes have any clean up at the end... TBD
        if __debug__:
            # Create code profiler
            self.code_profiler = cProfile.Profile()
            profile_prefix = "stimulation_profile"
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
        self.bayesian_estimator  = None
        self.graphical_interface = None

        # Put all the processes in a list so that we don't have to deal with
        # each of them by name when starting/stopping streaming.
        self.active_processes    = list()

        # Launch the main graphical interface as a widget
        self.setGeometry(100, 100, 900, 1200)

        """
        # The 2 lines below remove the CLOSE button on the window.
        # enable custom window hint
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)

        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        """

    def closeEvent(self, event):
        self.disconnectAndQuit()

    # Functions for saving data
    def saveFields(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def saveBayesianDecoding(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    # Functions for loading data
    def loadFields(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    ############################# STIMULATION TRIGGERS #############################
    # Set up the different stimulation methods here. The three different
    # methods that we plan to use right now are:
    #   1. Manual Trigger - Trigger for a fixed duration immediately after the
    #       menu is selected.
    #   2. Position Trigger - Allow the user to select a position and velocity
    #       cutoff within which stimulation will be activated.
    #   3. Ripple Trigger - The good old, trigger on Sharp-Wave ripples in the
    #   Hippocampus. It can be tricky to eliminate noise, whose broadband power
    #   can also be seen in the ripple band.
    #
    #   TODO: For ripple power, also incorporate a reference channel which DOES
    #       NOT HAVE SWRs
    ################################################################################

    def manualStimTrigger(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def positionStimTrigger(self, state):
        self.stim_mode_position.setChecked(not state)
        # QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def rippleStimTrigger(self, state):
        self.stim_mode_ripple.setChecked(not state)
        if state:
            self.ripple_trigger.disable()
        else:
            self.ripple_trigger.enable()

    ############################# SERIAL FUNCTIONALITY #############################
    # Add functions that let you access and test the serial port in a
    # convenient way. This allows you to safely enable/disable the serial port
    # and test the stimulating electrode's status by sending a single pulse OR
    # a series of pulses.
    ################################################################################

    # Set up the serial port
    def enableSerialPort(self, state):
        # TODO: To the information statement above, add a line telling which
        # port is currently being used.
        if state:
            QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Enabling serial port.')
            self.ripple_trigger.enableSerial()
        else:
            QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Disabling serial port.')
            self.ripple_trigger.disableSerial()

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

    # Show ripple stats (current as well as history) for tetrodes
    def showRippleStats(self):
        QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Functionality not implemented!')

    def disconnectAndQuit(self):
        QtHelperUtils.display_information(MODULE_IDENTIFIER + 'Quitting Program!')
        if self.graphical_interface is not None:
            self.graphical_interface.kill_gui()

        if self.data_streaming:
            self.stopThreads()

        try:
            self.sg_client.closeConnections()
        except Exception as err:
            print(MODULE_IDENTIFIER + "Unable to close connection to Trodes. Not that it won't throw seg fault in your face anyways!")
            print(err)

        print(MODULE_IDENTIFIER + "Program finished. Exiting.")
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
        output_menu = menu_bar.addMenu('&Output')
        plot_bayesian_estimate = output_menu.addAction('&Bayesian Estimate')
        plot_bayesian_estimate.setStatusTip('Plot bayesian estimate')
        plot_bayesian_estimate.triggered.connect(self.plotBayesianEstimate)

        print_ripple_stats = output_menu.addAction('&Ripple Statistics')
        print_ripple_stats.setStatusTip('Show ripple stats')
        print_ripple_stats.triggered.connect(self.showRippleStats)

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
        stim_mode_manual.triggered.connect(self.manualStimTrigger)
        stim_mode_manual.setShortcut('Ctrl+M')

        self.stim_mode_position = QAction('&Position', self, checkable=True)
        self.stim_mode_position.setStatusTip('Use position and velocity to simulate.')
        self.stim_mode_position.setChecked(DEFAULT_STIM_MODE_POSITION_ENABLED)
        self.stim_mode_position.triggered.connect(self.positionStimTrigger)

        self.stim_mode_ripple = QAction('&Ripple', self, checkable=True)
        self.stim_mode_ripple.setStatusTip('Stimulate on Sharp-Wave Ripples.')
        self.stim_mode_ripple.setChecked(DEFAULT_STIM_MODE_RIPPLE_ENABLED)
        self.stim_mode_ripple.triggered.connect(self.rippleStimTrigger)

        stimulation_menu.addAction(self.stim_mode_position)
        stimulation_menu.addAction(self.stim_mode_ripple)

    def getProcessingArgs(self):
        processing_args = list()
        processing_args.append("Local Field Potential (LFP)")
        processing_args.append("Spike Data")
        processing_args.append("Position Data")
        processing_args.append("Place Field")
        processing_args.append("Stimulation")
        user_choices = QtHelperUtils.CheckBoxWidget(processing_args, message="Select position processing options.").exec_()
        user_processing_choices = dict()
        user_processing_choices['lfp']      = DEFAULT_LFP_CHOICE
        user_processing_choices['spikes']   = DEFAULT_SPIKES_CHOICE
        user_processing_choices['position'] = DEFAULT_POSITION_CHOICE
        user_processing_choices['field']    = DEFAULT_FIELD_CHOICE
        user_processing_choices['stim']     = DEFAULT_STIMULATION_CHOICE
        if user_choices[0] == QMessageBox.Ok:
            if 0 in user_choices[1]:
                user_processing_choices['lfp'] = True
            else:
                user_processing_choices['lfp'] = False

            if 1 in user_choices[1]:
                user_processing_choices['spikes'] = True
            else:
                user_processing_choices['spikes'] = False

            if 2 in user_choices[1]:
                user_processing_choices['position'] = True
            else:
                user_processing_choices['position'] = False

            if 3 in user_choices[1]:
                user_processing_choices['field'] = True
            else:
                user_processing_choices['field'] = False

            if 4 in user_choices[1]:
                user_processing_choices['stim'] = True
            else:
                user_processing_choices['stim'] = False
        return user_processing_choices

    def connectSpikeGadgets(self):
        """
        Connect to Trodes client.
        """
        if self.sg_client is not None:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Already connected to Trodes! Try restarting to re-connect.')
            return

        try:
            self.sg_client = TrodesInterface.SGClient("ReplayInterruption")
        except Exception as err:
            QtHelperUtils.display_warning('Unable to connect to Trodes!')
            print(err)
            return
        if not self.cluster_identity_map:
            self.loadClusterFile()

        # Use preferences to selectively start the desired threads
        user_processing_choices = self.getProcessingArgs()
        try:
            if user_processing_choices['lfp']:
                print(MODULE_IDENTIFIER + "Starting LFP data Threads.")
                # LFP Threads
                self.initLFPThreads()

            if user_processing_choices['position']:
                # Position data
                print(MODULE_IDENTIFIER + "Starting Position data Threads.")
                self.position_estimator  = PositionAnalysis.PositionEstimator(self.sg_client)

            if user_processing_choices['spikes']:
                print(MODULE_IDENTIFIER + "Starting Spike data/processing Threads.")

                # Spike data
                self.initSpikeThreads()

                # Calibration data
                self.initCalibrationThreads()

            # Place fields depend on Spike and Position threads being present
            if user_processing_choices['position'] and user_processing_choices['spikes'] and user_processing_choices['field']:
                print(MODULE_IDENTIFIER + "Starting Place field processing Threads.")
                self.initPlaceFieldThreads()

            if user_processing_choices['lfp']:
                # Ripple triggered actions
                # This has to done after the threads above because this thread
                # write to the calibration plot thread after it has detected a
                # ripple.
                print(MODULE_IDENTIFIER + "Starting Sharp-Wave Ripple processing Threads.")
                self.initRippleTriggerThreads()

            if user_processing_choices['stim']:
                # Stimulation Threads
                print(MODULE_IDENTIFIER + "Starting Stimulation Threads.")
                self.initStimulationThreads()

            self.graphical_interface = Visualization.GraphicsManager((self.shared_raw_lfp_buffer,\
                    self.shared_ripple_buffer), (self.shared_calib_plot_means, self.shared_calib_plot_std_errs),\
                    self.spike_listener, self.position_estimator, self.place_field_handler, self.ripple_trigger,\
                    self.show_trigger, self.calib_trigger, self.shared_place_fields)

            # Load the tetrode and cluster information into the respective menus.
            self.graphical_interface.setTetrodeList(self.cluster_identity_map.keys())

            # Get a list of all the units that we have in the data-set
            session_unit_list = list()
            for tetrode in self.cluster_identity_map.keys():
                tetrode_units = self.cluster_identity_map[tetrode].values()
                session_unit_list.extend(tetrode_units)
            self.graphical_interface.setUnitList(session_unit_list)

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
        cluster_filename = "./config/full_config20190307.trodesClusters"
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
            """
            # Moving away from shutting individual threads down to stopping a list of specified threads.
            self.spike_listener.join()
            self.position_estimator.join()
            self.place_field_handler.join()
            self.lfp_listener.join()
            self.calib_plot.join()
            self.ripple_detector.join()
            self.ripple_trigger.join()
            """
            for requested_process in self.active_processes:
                requested_process.join()

        except Exception as err:
            logging.debug(MODULE_IDENTIFIER + "Caught Interrupt while exiting...")
            print(err)

    def startThreads(self):
        """
        Start all the processing thread.
        """
        try:
            """
            # This was the old way of dealing with each process individually - leading to BUGS!!!
            self.position_estimator.start()
            self.lfp_listener.start()
            self.ripple_detector.start()
            self.spike_listener.start()
            self.place_field_handler.start()
            self.ripple_trigger.start()
            self.calib_plot.start()
            """
            self.graphical_interface.start()
            for requested_process in self.active_processes:
                requested_process.start()

            if self.ripple_trigger is not None:
                # Select disruption mode
                self.rippleStimTrigger(not DEFAULT_STIM_MODE_RIPPLE_ENABLED)
                self.positionStimTrigger(not DEFAULT_STIM_MODE_POSITION_ENABLED)

        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to stream. Check connection to client!')
            print(err)
            return

        self.statusBar().showMessage('Streaming!')
        if __debug__:
            self.code_profiler.enable()

    def initRippleTriggerThreads(self):
        """
        Initialize threads dependent on ripple triggers (these need pretty much everything!)
        """
        try:
            self.shared_raw_lfp_buffer = RawArray(ctypes.c_double, self.n_tetrodes * RiD.LFP_BUFFER_LENGTH)
            self.shared_ripple_buffer = RawArray(ctypes.c_double, self.n_tetrodes * RiD.RIPPLE_POWER_BUFFER_LENGTH)
            self.ripple_detector = RippleAnalysis.RippleDetector(self.lfp_listener, self.calib_plot,\
                    trigger_condition=(self.trig_condition, self.show_trigger, self.calib_trigger),\
                    shared_buffers=(self.shared_raw_lfp_buffer, self.shared_ripple_buffer))
            self.active_processes.append(self.ripple_detector)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start ripple trigger threads(s).')
            print(err)
            return

    def initStimulationThreads(self):
        """
        Initialize threads for electrical/optical stimulation.
        """
        try:
            self.ripple_trigger = StimulationSynchronizer(self.trig_condition, self.spike_listener,\
                    self.position_estimator, self.place_field_handler, self.sg_client)
            self.active_processes.append(self.ripple_trigger)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start stimulation threads(s).')
            print(err)
            return


    def initLFPThreads(self):
        """
        Initialize the threads needed for LFP processing.
        """
        try:
            tetrode_argument = [str(tet) for tet in self.tetrodes_of_interest]
            self.lfp_listener = RippleAnalysis.LFPListener(self.sg_client, tetrode_argument)
            self.active_processes.append(self.lfp_listener)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start LFP thread(s).')
            print(err)
            return

    def initSpikeThreads(self):
        """
        Initialize thread needed for spike processing
        """
        try:
            self.spike_listener = SpikeAnalysis.SpikeDetector(self.sg_client, self.cluster_identity_map)

            # Update the active process list
            self.active_processes.append(self.spike_listener)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start spike thread(s).')
            print(err)
            return

    def initPlaceFieldThreads(self):
        """
        Initialize thread needed for building and visualizing place fields.
        """
        try:
            self.shared_place_fields = RawArray(ctypes.c_double, self.n_units * PositionAnalysis.N_POSITION_BINS[0] * \
                    PositionAnalysis.N_POSITION_BINS[1])
            self.place_field_handler = SpikeAnalysis.PlaceFieldHandler(self.position_estimator,\
                    self.spike_listener, self.shared_place_fields)
            # self.bayesian_estimator = PositionDecoding.BayesianEstimator(self.spike_listener, \
            #     self.place_field_handler, self.shared_place_fields)

            # Update the active process list
            self.active_processes.append(self.place_field_handler)
            # self.active_processes.append(self.bayesian_estimator)
        except Exception as err:
            QtHelperUtils.display_warning(MODULE_IDENTIFIER + 'Unable to start Place Field thread(s).')
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
            self.active_processes.append(self.calib_plot)
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
        """
        logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
                level=logging.INFO, datefmt="%H:%M:%S")
        """
        logging.basicConfig(filename=log_filename, format="%(asctime)s.%(msecs)03d:%(message)s", \
                level=logging.DEBUG, datefmt="%H:%M:%S")
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
