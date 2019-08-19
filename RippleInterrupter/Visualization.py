"""
Visualization of various measures
"""

from collections import deque
from multiprocessing import Process, Event
import tkinter
import time
import numpy as np
from datetime import datetime
import threading
from scipy.ndimage.filters import gaussian_filter
import logging

# Matplotlib in Qt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import gridspec
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import matplotlib.animation as animation

# Creating windows using PyQt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QDialog, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QPushButton, QSlider, QRadioButton, QLabel, QInputDialog, QTextEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# Local Imports
import Configuration
import RippleAnalysis
import PositionAnalysis
import RippleDefinitions as RiD

MODULE_IDENTIFIER = "[GraphicsHandler] "
ANIMATION_INTERVAL = 20

def normalizeData(in_data):
    # TODO: Might need tiling of data if there are multiple dimensions
    data_mean = np.mean(in_data, axis=0)
    data_std  = np.std(in_data, axis=0)
    norm_data = np.divide((in_data - data_mean), data_std)
    return (norm_data, data_mean, data_std)

def animateLFP(timestamps, lfp, raw_ripple, ripple_power, frame_size, statistic=None):
    """
    Animate a given LFP by plotting a fixed size sliding frame.

    :timestamps: time-points for the LFP data
    :lfp: LFP (Raw/Filtered) for a single tetrode
    :raw_ripple: LFP Passed through a ripple filter
    :ripple_power: Ripple power calculated over a moving winow centered at all
        the data points.
    :frame_size: Size of the frame that should be seen at once
    :statistic: Function handle that should be applied to the data to generate
        a scalar quantity that can also be plotted!
    """

    # Turn interactive plotting off. It messes up animation
    plt.ioff()

    # Change this to '3d' if the need every arises for a multi-dimensional plot
    lfp_fig   = plt.figure()
    plot_axes = plt.axes(projection=None)

    # Start with an empty plot, it can be then updated by animation functions
    # NOTE: The way frame is accessed in animation internals forces us to
    # make this an array if nothing else is being passed in. Having text
    # removes this requirement.
    lfp_frame,   = plot_axes.plot([], [], animated=True)
    r_raw_frame, = plot_axes.plot([], [], animated=True)
    r_pow_frame, = plot_axes.plot([], [], animated=True)
    txt_template = 't = %.2fs'
    lfp_measure  = plot_axes.text(0.5, 0.09, '', transform=plot_axes.transAxes)

    # Local functions for setting up animation frames and cycling through them
    def _nextAnimFrame(step=0):
        """
        # Making sure that the step index and data are coming in properly
        print(step)
        print(lfp[step])
        """
        lfp_frame.set_data(timestamps[step:step+frame_size], lfp[step:step+frame_size])
        r_raw_frame.set_data(timestamps[step:step+frame_size], raw_ripple[step:step+frame_size])
        r_pow_frame.set_data(timestamps[step:step+frame_size], ripple_power[step:step+frame_size])
        lfp_measure.set_text(txt_template % timestamps[step])
        # Updating the limits is needed still so that the correct range of data
        # is displayed! It doesn't update the axis labels though - That's a
        # different ballgame!
        plot_axes.set_xlim(timestamps[step], timestamps[step+frame_size])
        return lfp_frame, r_raw_frame, r_pow_frame, lfp_measure

    def _initAnimFrame():
        # NOTE: Init function called twice! I have seen this before but still
        # don't understand why it works this way!
        # print("Initializing animation frame...")
        plot_axes.set_xlabel('Time (s)')
        plot_axes.set_ylabel('EEG (uV)')
        plot_axes.set_ylim(min(lfp), max(lfp))
        plot_axes.set_xlim(timestamps[0], timestamps[frame_size])
        plot_axes.grid(True)
        return _nextAnimFrame()

    n_frames = len(timestamps) - frame_size
    lfp_anim = animation.FuncAnimation(lfp_fig, _nextAnimFrame, np.arange(0, n_frames), \
            init_func=_initAnimFrame, interval=RiD.LFP_ANIMATION_INTERVAL, \
            blit=True, repeat=False)
    plt.figure(lfp_fig.number)

    # Make the filtered ripple thinner
    r_raw_frame.set_linewidth(0.5)
    plt.show(plot_axes)

def visualizeLFP(timestamps, raw_lfp_buffer, ripple_power, ripple_filtered_lfp, \
        ripple_events=None, do_animation=False):
    # Normalize both EEG and Ripple power so that they can be visualized together.
    norm_lfp, lfp_mean, lfp_std = normalizeData(raw_lfp_buffer[0,:])
    norm_ripple_power, power_mean, power_std = normalizeData(ripple_power[0,:])
    norm_raw_ripple, ripple_mean, ripple_std = normalizeData(ripple_filtered_lfp[0,:])

    print("Ripple Statistics...")
    print("Mean: %.2f"%power_mean)
    print("Std: %.2f"%power_std)

    # Plots - Pick a tetrode to plot the data from
    # Static plots
    plt.ion()
    n_tetrodes = 1
    for tetrode_idx in range(n_tetrodes):
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(timestamps, norm_lfp)
        if ripple_events is not None:
            ax1.scatter(ripple_events, np.zeros(len(ripple_events)), c="r")
        ax1.grid(True)
        ax2.plot(timestamps, norm_raw_ripple)
        ax2.plot(timestamps, norm_ripple_power)
        ax2.grid(True)
        plt.show()

    """
    # Plot a histogram of the LFP power
    plt.figure()
    hist_axes = plt.axes()
    plt.hist(norm_ripple_power, bins=RiD.N_POWER_BINS, density=True)
    plt.grid(True)
    hist_axes.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.show()
    """

    if do_animation:
        # Animation
        wait_for_user_input = input('Press ENTER to continue, or Q to ABORT.')
        if (wait_for_user_input == 'Q'):
            return
        animateLFP(timestamps, norm_lfp, norm_raw_ripple, norm_ripple_power, 400)
    return (power_mean, power_std)

class GraphicsManager(Process):
    """
    Process for managing visualization and graphics
    """

    __N_POSITION_ELEMENTS_TO_PLOT = 100
    __N_SPIKES_TO_PLOT = 2000
    __N_ANIMATION_FRAMES = 50000
    __PLACE_FIELD_REFRESH_RATE = 1
    __PLOT_REFRESH_RATE = 0.05
    __PEAK_LFP_AMPLITUDE = 1000
    __CLUSTERS_TO_PLOT = []
    __N_SUBPLOT_COLS = int(3)
    __MAX_FIRING_RATE = 15.0
    __RIPPLE_DETECTION_TIMEOUT = 0.5
    __RIPPLE_SMOOTHING_WINDOW = 2

    def __init__(self, ripple_buffers, calib_plot_buffers, spike_listener, position_estimator, \
            place_field_handler, ripple_trigger_thread, ripple_trigger_condition, calib_trigger_condition, \
            shared_place_fields, clusters=None):
        """
        Graphical Manager for all the processes
        :ripple_analyzer: Process/Thread listening to LFP and detecting ripples
        :spike_listener: Thread listening to incoming raw spike stream from trodes.
        :position_estimator: Thread listening to position data from camera stream
        :place_field_handler: Process constructing place fields
        :ripple_trigger_condition: TODO
        :clusters: User specified cluster indices that we should be looking at.
        """
        Process.__init__(self)

        # Graphics windows
        self.widget  = QDialog()
        self.figure  = Figure(figsize=(12,16))
        self.canvas  = FigureCanvas(self.figure)

        # Count the number of requested features
        n_features_to_show = 0
        if ripple_buffers[0] is not None:
            # Ripple data needs to be shown
            n_features_to_show += 1

        if calib_plot_buffers[0] is not None:
            n_features_to_show += 1

        if position_estimator is not None:
            n_features_to_show += 1

        if place_field_handler is not None:
            n_features_to_show += 1

        # The layout of the application can be different based on what features
        # are being requested
        if n_features_to_show == 1:
            plot_grid = gridspec.GridSpec(1, 1)
        elif n_features_to_show == 2:
            plot_grid = gridspec.GridSpec(1, 2)
        elif n_features_to_show == 3:
            plot_grid = gridspec.GridSpec(1, 3)
        elif n_features_to_show == 4:
            plot_grid = gridspec.GridSpec(2, 2)

        self.toolbar = NavigationToolbar(self.canvas, self.widget)
        current_grid_place = 0
        if ripple_buffers[0] is not None:
            self._rd_ax = self.figure.add_subplot(plot_grid[current_grid_place])
            current_grid_place += 1
        else:
            self._rd_ax = None

        if calib_plot_buffers[0] is not None:
            self._cp_ax = self.figure.add_subplot(plot_grid[current_grid_place])
            current_grid_place += 1
        else:
            self._cp_ax = None

        if position_estimator is not None:
            self._spk_pos_ax = self.figure.add_subplot(plot_grid[current_grid_place])
            current_grid_place += 1
        else:
            self._spk_pos_ax = None

        if place_field_handler is not None:
            self._pf_ax = self.figure.add_subplot(plot_grid[current_grid_place])
            current_grid_place += 1
        else:
            self._pf_ax = None

        # Selecting individual units
        self.unit_selection = QComboBox()
        self.user_message = QTextEdit()
        self.user_message.resize(300, 100)
        self.log_message = QPushButton('Log')
        self.clear_message = QPushButton('Clear')
        self.log_message.clicked.connect(self.LogUserMessage)
        self.clear_message.clicked.connect(self.ClearUserMessage)

        # self.unit_selection.currentIndexChanged.connect(self.refresh)
        # Add next and prev buttons to look at individual cells.
        self.next_unit_button = QPushButton('Next')
        self.next_unit_button.clicked.connect(self.NextUnit)
        self.prev_unit_button = QPushButton('Prev')
        self.prev_unit_button.clicked.connect(self.PrevUnit)

        # Selecting individual tetrodes
        self.tetrode_selection = QComboBox()
        # self.tetrode_selection.currentIndexChanged.connect(self.refresh)
        # Add next and prev buttons to look at individual cells.
        self.next_tet_button = QPushButton('Next')
        self.next_tet_button.clicked.connect(self.NextTetrode)
        self.prev_tet_button = QPushButton('Prev')
        self.prev_tet_button.clicked.connect(self.PrevTetrode)

        self._keep_running = Event()
        self._spike_listener = spike_listener
        self._position_estimator = position_estimator
        self._place_field_handler = place_field_handler
        self._ripple_trigger_thread = ripple_trigger_thread
        self._ripple_trigger_condition = ripple_trigger_condition
        self._calib_trigger_condition = calib_trigger_condition

        # Look at spikes if we are going to be getting them
        if (self._spike_listener is not None):
            self._n_total_clusters = self._spike_listener.get_n_clusters()
            self._n_tetrodes = len(self._spike_listener.get_tetrodes())
            if clusters is None:
                # These are the clusters we are going to plot
                self._n_clusters = len(self.__CLUSTERS_TO_PLOT)
                self._clusters = self.__CLUSTERS_TO_PLOT
            else:
                self._n_clusters = len(clusters)
                self._clusters = clusters
        else:
            self._n_clusters = 0
            self._clusters = list()
            self._n_total_clusters = 0
            self._n_tetrodes = 0

        self._cluster_colormap = colormap.magma(np.linspace(0, 1, self._n_clusters))
        # Create a list of threads depending on the requeseted features.
        self._thread_list = list()

        # Enable Ripple Buffer and corresponding thread if requested
        if ripple_buffers[0] is not None:
            # If we are not getting spike data, then this needs to be updated using the LFP BUFFER
            if self._n_tetrodes == 0:
                self._n_tetrodes = int(len(ripple_buffers[0])/RiD.LFP_BUFFER_LENGTH)
            self._shared_raw_lfp_buffer = np.reshape(np.frombuffer(ripple_buffers[0], dtype='double'), \
                    (self._n_tetrodes, RiD.LFP_BUFFER_LENGTH))
            self._shared_ripple_power_buffer = np.reshape(np.frombuffer(ripple_buffers[1], dtype='double'), \
                    (self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH))
            self._thread_list.append(threading.Thread(name="RippleFrameFetcher", daemon=True, \
                    target=self.fetch_incident_ripple))
            logging.info(MODULE_IDENTIFIER + "Added Ripple threads to Graphics pipeline.")
        else:
            self._shared_raw_lfp_buffer = None
            self._shared_ripple_power_buffer = None

        # Enable position data and thread if requested
        if self._position_estimator is not None:
            self._position_buffer = self._position_estimator.get_position_buffer_connection()
            self._thread_list.append(threading.Thread(name="PositionFetcher", daemon=True, \
                    target=self.fetch_position_and_update_frames))
            logging.info(MODULE_IDENTIFIER + "Added Position threads to Graphics pipeline.")

        # Enable place field handler if requested
        if (self._place_field_handler is not None) and (shared_place_fields is not None):
            self._spike_buffer = self._place_field_handler.get_spike_place_buffer_connection(self._clusters)
            self._thread_list.append(threading.Thread(name="SpikeFetcher", daemon=True, \
                    target=self.fetch_spikes_and_update_frames))
            self._thread_list.append(threading.Thread(name="PlaceFieldFetched", daemon=True, \
                    target=self.fetch_place_fields))
            self._shared_place_fields = np.reshape(np.frombuffer(shared_place_fields, dtype='double'), \
                    (self._n_total_clusters, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))
            logging.info(MODULE_IDENTIFIER + "Added Spike/Place-Field threads to Graphics pipeline.")
        else:
            self._shared_place_fields = None


        # Enable spike calibration plots if requested
        self._calib_lock = threading.Lock()
        if calib_plot_buffers[0] is not None: 
            self._shared_calib_plot_means = np.reshape(np.frombuffer(calib_plot_buffers[0], dtype='double'), \
                    (RiD.CALIB_PLOT_BUFFER_LENGTH))
            self._shared_calib_plot_std_errs = np.reshape(np.frombuffer(calib_plot_buffers[1], dtype='double'), \
                    (RiD.CALIB_PLOT_BUFFER_LENGTH))
            self._thread_list.append(threading.Thread(name="CalibPlotFetcher", daemon=True, \
                    target=self.fetch_calibration_plot))
            logging.info(MODULE_IDENTIFIER + "Added Calibration threads to Graphics pipeline.")
        else:
            self._shared_calib_plot_means = None
            self._shared_calib_plot_std_errs = None

        # Local copies of the shared data that can be used at a leisurely pace
        self._lfp_lock = threading.Lock()
        self._lfp_tpts = np.linspace(-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME, RiD.LFP_BUFFER_LENGTH)
        self._ripple_power_tpts = np.linspace(-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME, RiD.RIPPLE_POWER_BUFFER_LENGTH)
        self._local_lfp_buffer = np.zeros((self._n_tetrodes, RiD.LFP_BUFFER_LENGTH), dtype='double')
        self._local_ripple_power_buffer = np.zeros((self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH), dtype='double')

        self._pf_lock = threading.Lock()
        self._most_recent_pf = np.zeros((PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]), \
                dtype='float')

        self._spk_cnt_tpts = np.linspace(0, RiD.CALIB_PLOT_BUFFER_TIME, RiD.CALIB_PLOT_BUFFER_LENGTH)
        self._local_spk_cnt_buffer = np.zeros((RiD.CALIB_PLOT_BUFFER_LENGTH), dtype='double')
        self._local_spk_cnt_stderr_buffer = np.zeros((RiD.CALIB_PLOT_BUFFER_LENGTH), dtype='double')

        # Automatically keep only a fixed number of entries in this buffer... Useful for plotting
        self._pos_timestamps = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_lock = threading.Lock()
        self._pos_x = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_y = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._speed = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)

        # Maintain a separate deque for each cluster to plot
        self._spike_lock = threading.Lock()
        self.initSpikeDeque()

        # Figure/Animation element. So far the following have been included
        # Ripple detection
        # Place Fields
        # Position/Spikes overalaid
        self._rd_frame = list()
        self._spk_pos_frame = list()
        self._pf_frame = list()
        self._cp_frame = list()
        self._anim_objs = list()

        logging.info(MODULE_IDENTIFIER + "Graphics interface started.")
        self.setLayout()
        self.clearAxes()

        # Start the animation for Spike-Position figure, place field figure
        self.initialize_ripple_detection_fig()
        self.initialize_calib_plot_fig()
        self.initialize_spike_pos_fig()
        self.initialize_place_field_fig()

        self.pauseAnimation()
        self._keep_running.set()
        for p__thread in self._thread_list:
            p__thread.start()
        print(MODULE_IDENTIFIER + 'Animation plots initialized.')


    def initSpikeDeque(self):
        self._spk_pos_x = []
        self._spk_pos_y = []
        for cl_idx in range(self._n_clusters):
            self._spk_pos_x.append(deque([], self.__N_SPIKES_TO_PLOT))
            self._spk_pos_y.append(deque([], self.__N_SPIKES_TO_PLOT))

    def setLayout(self):
        parent_layout_box = QVBoxLayout()
        parent_layout_box.addWidget(self.toolbar)
        parent_layout_box.addWidget(self.canvas)
        parent_layout_box.addStretch(1)

        # Controls for looking at individual units
        vbox_unit_buttons = QVBoxLayout()
        vbox_unit_buttons.addWidget(self.unit_selection)
        vbox_unit_buttons.addWidget(self.next_unit_button)
        vbox_unit_buttons.addWidget(self.prev_unit_button)

        # Controls for looking at individual tetrodes for LFP
        vbox_tetrode_buttons = QVBoxLayout()
        vbox_tetrode_buttons.addWidget(self.tetrode_selection)
        vbox_tetrode_buttons.addWidget(self.next_tet_button)
        vbox_tetrode_buttons.addWidget(self.prev_tet_button)

        # Add a block for user to add comments
        message_button_box = QHBoxLayout()
        message_button_box.addStretch(1)
        message_button_box.addWidget(self.log_message)
        message_button_box.addWidget(self.clear_message)
        message_button_box.addStretch(1)

        vbox_user_message = QVBoxLayout()
        vbox_user_message.addWidget(self.user_message)
        vbox_user_message.addStretch(1)
        vbox_user_message.addLayout(message_button_box)

        # Put the tetrode and unit buttons together
        hbox_unit_and_tet_controls = QHBoxLayout()
        hbox_unit_and_tet_controls.addLayout(vbox_user_message)
        hbox_unit_and_tet_controls.addLayout(vbox_unit_buttons)
        hbox_unit_and_tet_controls.addLayout(vbox_tetrode_buttons)

        parent_layout_box.addLayout(hbox_unit_and_tet_controls)
        QDialog.setLayout(self.widget, parent_layout_box)

    def setClusterIdentities(self, cluster_identity_map):
        # Take a cluster identity map and use it to populate the tetrodes and units.
        pass

    def setUnitList(self, unit_list):
        # Take the list of units and set them as the current list of units to be looked at.
        # print(unit_list)
        unit_id_strings = [str(unit_id) for unit_id in unit_list]
        self.unit_selection.addItems(unit_id_strings)

        # Update the cluster information
        self._n_clusters = len(unit_list)
        self._clusters = unit_list
        self.initSpikeDeque()

    def setTetrodeList(self, tetrode_list):
        tetrode_id_strings = [str(tet_id) for tet_id in tetrode_list]
        self.tetrode_selection.addItems(tetrode_id_strings)

    def LogUserMessage(self):
        pass

    def ClearUserMessage(self):
        pass

    # Saving Images
    def saveDisplay(self):
        if self.figure is None:
            return

        # Create a filename
        save_file_name = time.strftime("T" + str(self.tetrode_selection.currentText()) + "_%Y%m%d_%H%M%S.png") 
        save_success = False
        try:
            self.figure.savefig(save_file_name)
            save_success = True
        except Exception as err:
            print(MODULE_IDENTIFIER + "Unable to save current display.")
            print(err)
        return save_success

    # Saving Videos
    def recordDisplay(self):
        pass

    def NextUnit(self):
        current_unit = self.unit_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Unit: %d'%current_unit)
        if current_unit < self.unit_selection.count()-1:
            self.unit_selection.setCurrentIndex(current_unit+1)
        print(MODULE_IDENTIFIER + "%d spikes received for current unit"%len(self._spk_pos_x[current_unit]))

    def PrevUnit(self):
        current_unit = self.unit_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Unit: %d'%current_unit)
        if current_unit > 0:
            self.unit_selection.setCurrentIndex(current_unit-1)
        print(MODULE_IDENTIFIER + "%d spikes received for current unit"%len(self._spk_pos_x[current_unit]))

    def NextTetrode(self):
        current_tet = self.tetrode_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Tetrode: %d'%current_tet)
        with self._lfp_lock:
            if current_tet < self.tetrode_selection.count()-1:
                self.tetrode_selection.setCurrentIndex(current_tet+1)

    def PrevTetrode(self):
        current_tet = self.tetrode_selection.currentIndex()
        # print(MODULE_IDENTIFIER + 'Current Tetrode: %d'%current_tet)
        with self._lfp_lock:
            if current_tet > 0:
                self.tetrode_selection.setCurrentIndex(current_tet-1)

    def clearAxes(self):
        # Ripple detection axis
        if self._rd_ax is not None:
            self._rd_ax.cla()
            self._rd_ax.set_xlabel("Time (s)")
            self._rd_ax.set_ylabel("Ripple Power (STD)")
            self._rd_ax.set_xlim((-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME))
            self._rd_ax.set_ylim((-1.0, 1.6*RiD.RIPPLE_POWER_THRESHOLD))
            self._rd_ax.grid(True)

        # Calibration plot
        if self._cp_ax is not None:
            self._cp_ax.cla()
            self._cp_ax.set_xlabel("Time (s)")
            self._cp_ax.set_ylabel("Spike Rate (spks/5ms)")
            self._cp_ax.set_xlim((-0.5 * RiD.LFP_BUFFER_TIME, 0.5 * RiD.LFP_BUFFER_TIME))
            self._cp_ax.set_ylim((0.0, 20.0))
            self._cp_ax.grid(True)

        # Place field
        if self._pf_ax is not None:
            self._pf_ax.cla()
            self._pf_ax.set_xlabel("x (bin)")
            self._pf_ax.set_ylabel("y (bin)")
            self._pf_ax.set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
            self._pf_ax.set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
            self._pf_ax.grid(True)

        # Spikes (from a single cell) and position
        if self._spk_pos_ax is not None:
            self._spk_pos_ax.cla()
            self._spk_pos_ax.set_xlabel("x (bin)")
            self._spk_pos_ax.set_ylabel("y (bin)")
            self._spk_pos_ax.set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
            self._spk_pos_ax.set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
            self._spk_pos_ax.grid(True)

        self.canvas.draw()

    def kill_gui(self):
        self._keep_running.clear()
        
    def update_ripple_detection_frame(self, step=0):
        """
        Function used to show a ripple frame whenever a ripple is trigerred.
        This is a little different from the other frame update functions as it
        does not continuously update the frame but only when a ripple is triggerred.
        """

        # NOTE: This call blocks access to ripple_trigger_condition for
        # __RIPPLE_DETECTION_TIMEOUT, which could be a long while. Don't let
        # this block any important functionality.
        with self._lfp_lock:
            current_tetrode_selection = self.tetrode_selection.currentIndex()
            self._rd_frame[0].set_data(self._lfp_tpts, 3.0 + self._local_lfp_buffer[current_tetrode_selection,:]/self.__PEAK_LFP_AMPLITUDE)

            # Smooth out the ripple power
            """
            smoothed_ripple_power = \
                    gaussian_filter(self._local_ripple_power_buffer[current_tetrode_selection,:], \
                    self.__RIPPLE_SMOOTHING_WINDOW)
            """
            smoothed_ripple_power = self._local_ripple_power_buffer[current_tetrode_selection,:]
            self._rd_frame[1].set_data(self._ripple_power_tpts, smoothed_ripple_power)
        return self._rd_frame

    def update_calib_plot_frame(self, step=0):
        """
        Function used to show a ripple frame whenever a ripple is trigerred.
        This is a little different from the other frame update functions as it
        does not continuously update the frame but only when a ripple is triggerred.
        """
        
        # NOTE: This call blocks access to ripple_trigger_condition for
        # __RIPPLE_DETECTION_TIMEOUT, which could be a long while. Don't let
        # this block any important functionality.
        with self._calib_lock:
            self._cp_frame[0].set_data(self._spk_cnt_tpts, self._local_spk_cnt_buffer)
            self._cp_frame[1].set_data(self._spk_cnt_tpts, self._local_spk_cnt_buffer + self._local_spk_cnt_stderr_buffer)
            self._cp_frame[2].set_data(self._spk_cnt_tpts, self._local_spk_cnt_buffer - self._local_spk_cnt_stderr_buffer)
        return self._cp_frame

    def update_position_and_spike_frame(self, step=0):
        """
        Function used for animating the current position of the animal.
        """
        cl_idx = max(self.unit_selection.currentIndex(), 0)
        with self._spike_lock:
            self._spk_pos_frame[0].set_data((self._spk_pos_x[cl_idx], self._spk_pos_y[cl_idx]))

        with self._pos_lock:
            self._spk_pos_frame[-2].set_data((self._pos_x, self._pos_y))
            if len(self._speed) > 0:
                self._spk_pos_frame[-1].set_text('speed = %.2fcm/s'%self._speed[-1])
        return self._spk_pos_frame

    def update_place_field_frame(self, step=0):
        """
        Function used for animating the place field for a particular spike cluster.
        TODO: Utility to be expanded to multiple clusters in the future.

        :step: Animation iteration
        :returns: Animation frames to be plotted.
        """
        # print("Peak FR: %.2f, Mean FR: %.2f"%(np.max(self._most_recent_pf), np.mean(self._most_recent_pf)))
        # print("Min FR: %.2f, Max FR: %.2f"%(np.min(self._most_recent_pf), np.max(self._most_recent_pf)))
        # min_fr = np.min(self._most_recent_pf)
        # max_fr = np.max(self._most_recent_pf)
        with self._pf_lock:
            self._pf_frame[0].set_array(self._most_recent_pf.T)
        return self._pf_frame
        
    def fetch_incident_ripple(self):
        """
        Fetch raw LFP data and ripple power data.
        """
        logging.info(MODULE_IDENTIFIER + "Ripple frame pipe opened.")
        while self._keep_running.is_set():
            with self._ripple_trigger_condition:
                ripple_triggered = self._ripple_trigger_condition.wait(self.__RIPPLE_DETECTION_TIMEOUT)

            if ripple_triggered:
                with self._lfp_lock:
                    np.copyto(self._local_lfp_buffer, self._shared_raw_lfp_buffer)
                    np.copyto(self._local_ripple_power_buffer, self._shared_ripple_power_buffer)
                logging.info(MODULE_IDENTIFIER + "Peak ripple power in frame %.2f"%np.max(self._shared_ripple_power_buffer))
            else:
                time.sleep(0.01)
        logging.info(MODULE_IDENTIFIER + "Ripple frame pipe closed.")

    def fetch_calibration_plot(self):
        """
        Fetch raw LFP data and ripple power data.
        """
        logging.info(MODULE_IDENTIFIER + "Calibration pipe opened.")
        while self._keep_running.is_set():
            with self._calib_trigger_condition:
                ripple_triggered = self._calib_trigger_condition.wait(self.__RIPPLE_DETECTION_TIMEOUT)

            if ripple_triggered:
                with self._calib_lock:
                    np.copyto(self._local_spk_cnt_buffer, self._shared_calib_plot_means)
                    np.copyto(self._local_spk_cnt_stderr_buffer, self._shared_calib_plot_std_errs)
            else:
                time.sleep(0.01)
        logging.info(MODULE_IDENTIFIER + "Calibration pipe closed.")

    def fetch_place_fields(self):
        """
        Fetch place field data from place field handler.
        """
        logging.info(MODULE_IDENTIFIER + "Place Field pipe opened.")
        while self._keep_running.is_set():
            time.sleep(self.__PLACE_FIELD_REFRESH_RATE)
            # Request place field handler to pause place field calculation
            # while we fetch the data
            self._place_field_handler.submit_immediate_request()
            with self._pf_lock:
                # Uncomment this line to get an average of all the place fields
                # np.mean(self._shared_place_fields, out=self._most_recent_pf, axis=0)

                # Uncomment to look at the place field of the selected unit
                np.copyto(self._most_recent_pf, self._shared_place_fields[self.unit_selection.currentIndex(), :, :])
            # Release the request that paused place field computation
            self._place_field_handler.end_immediate_request()
        logging.info(MODULE_IDENTIFIER + "Place Field pipe closed.")

    def fetch_spikes_and_update_frames(self):
        logging.info(MODULE_IDENTIFIER + "Spike pipe opened.")
        while self._keep_running.is_set():
            if self._spike_buffer.poll():
                spike_data = self._spike_buffer.recv()
                # TODO: This is a little inefficient. For every spike we get,
                # we check to see if it is in the clusters of interest and then
                # find its  
                if spike_data[0] in self._clusters:
                    data_idx = self._clusters.index(spike_data[0])
                    with self._spike_lock:
                        self._spk_pos_x[data_idx].append(spike_data[1])
                        self._spk_pos_y[data_idx].append(spike_data[2])
                logging.debug(MODULE_IDENTIFIER + "Fetched spike from cluster: %d, in bin (%d, %d). TS: %d"%spike_data)
            else:
                time.sleep(self.__PLOT_REFRESH_RATE)
        logging.info(MODULE_IDENTIFIER + "Spike pipe closed.")

    def fetch_position_and_update_frames(self):
        logging.info(MODULE_IDENTIFIER + "Position pipe opened.")
        while self._keep_running.is_set():
            if self._position_buffer.poll():
                position_data = self._position_buffer.recv()
                with self._pos_lock:
                    self._pos_timestamps.append(position_data[0])
                    self._pos_x.append(position_data[1])
                    self._pos_y.append(position_data[2])
                    self._speed.append(position_data[3])
                # print(self)
                # print(self._pos_x)
                # print(self._pos_y)
                logging.debug(MODULE_IDENTIFIER + "Fetched Position data... (%d, %d), v: %.2fcm/s"% \
                      (position_data[1],position_data[2], position_data[3]))
            else:
                time.sleep(self.__PLOT_REFRESH_RATE)
        logging.info(MODULE_IDENTIFIER + "Position pipe closed.")

    def pauseAnimation(self):
        """
        Pause all animation sources.
        """
        for ao in self._anim_objs:
            ao.event_source.stop()
    
    def playAnimation(self):
        """
        Play all animation sources.
        """
        for ao in self._anim_objs:
            ao.event_source.start()

    def initialize_ripple_detection_fig(self):
        """
        Initialize figure window for showing raw LFP and ripple power.
        :returns: TODO
        """
        if self._rd_ax is None:
            return

        lfp_frame, = self._rd_ax.plot([], [], animated=True)
        ripple_power_frame, = self._rd_ax.plot([], [], animated=True)
        self._rd_ax.legend((lfp_frame, ripple_power_frame), ('Raw LFP', 'Ripple Power'))
        self._rd_frame.append(lfp_frame)
        self._rd_frame.append(ripple_power_frame)

        # Create animation object for showing the EEG
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_ripple_detection_frame, frames=np.arange(self.__N_ANIMATION_FRAMES), \
                interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Ripple detection frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_calib_plot_fig(self):
        """
        Initialize figure window for showing raw LFP and ripple power.
        :returns: TODO
        """
        if self._cp_ax is None:
            return

        spk_cnt_frame, = self._cp_ax.plot([], [], animated=True)
        spk_cnt_plus_sterr_frame, = self._cp_ax.plot([], [], animated=True)
        spk_cnt_minus_sterr_frame, = self._cp_ax.plot([], [], animated=True)
        self._cp_frame.append(spk_cnt_frame)
        self._cp_frame.append(spk_cnt_plus_sterr_frame)
        self._cp_frame.append(spk_cnt_minus_sterr_frame)

        # Create animation object for showing the EEG
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_calib_plot_frame, frames=np.arange(self.__N_ANIMATION_FRAMES), \
                interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Spike calibration frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_spike_pos_fig(self):
        """
        Initialize figure window for showing spikes overlaid on position
        """
        if self._spk_pos_ax is None:
            return

        spk_frame, = self._spk_pos_ax.plot([], [], linestyle='None', marker='o', alpha=0.4, animated=True)
        pos_frame, = self._spk_pos_ax.plot([], [], animated=True)
        vel_frame  = self._spk_pos_ax.text(0.5 * PositionAnalysis.N_POSITION_BINS[0], \
                0.02 * PositionAnalysis.N_POSITION_BINS[1], 'speed = 0cm/s')
        self._spk_pos_frame.append(spk_frame)
        self._spk_pos_frame.append(pos_frame)
        self._spk_pos_frame.append(vel_frame)

        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_position_and_spike_frame, \
                frames=np.arange(self.__N_ANIMATION_FRAMES), interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Spike-Position frame created!')
        self._anim_objs.append(anim_obj)

    def initialize_place_field_fig(self):
        """
        Initialize figure window for dynamically showing place fields.
        """
        if self._pf_ax is None:
            return

        pf_heatmap = self._pf_ax.imshow(np.zeros((PositionAnalysis.N_POSITION_BINS[0], \
                PositionAnalysis.N_POSITION_BINS[1]), dtype='float'), vmin=0, \
                vmax=self.__MAX_FIRING_RATE, animated=True)
        self.figure.colorbar(pf_heatmap)
        self._pf_frame.append(pf_heatmap)
        anim_obj = animation.FuncAnimation(self.canvas.figure, self.update_place_field_frame, \
                frames=np.arange(self.__N_ANIMATION_FRAMES), interval=ANIMATION_INTERVAL, blit=True, repeat=True)
        logging.info(MODULE_IDENTIFIER + 'Place field frame created!')
        self._anim_objs.append(anim_obj)

    def run(self):
        """
        Start a GUI, launch all the graphics windows that have been requested
        in separate threads.
        """

        """
        self._keep_running.set()

        for p__thread in self._thread_list:
            p__thread.start()
        """

        # Start all the animation sources.
        self.playAnimation()

        # This is a blocking command... After you exit this, everything will end.
        while self._keep_running.is_set():
            time.sleep(1.0)

        # Stop all the animation sources.
        self.pauseAnimation()

        # Join all the fetcher threads.
        for p__thread in self._thread_list:
            p__thread.join()
        logging.info(MODULE_IDENTIFIER + "Terminated GUI and display pipes")
