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
import logging

# Matplotlib in Qt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import gridspec
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import matplotlib.animation as animation

# Creating windows using PyQt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QDialog, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QPushButton, QSlider, QRadioButton, QLabel, QInputDialog
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox

# Local Imports
import Configuration
import RippleAnalysis
import PositionAnalysis
import RippleDefinitions as RiD

MODULE_IDENTIFIER = "[GraphicsHandler] "

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
    __N_ANIMATION_FRAMES = 5000
    __PLACE_FIELD_REFRESH_RATE = 1
    __PLOT_REFRESH_RATE = 0.05
    __PEAK_LFP_AMPLITUDE = 3000
    __CLUSTERS_TO_PLOT = Configuration.EXPERIMENT_DAY_20190307__ALL_INTERESTING_CLUSTERS
    __N_SUBPLOT_COLS = int(3)
    __N_SUBPLOT_ROWS = int(1)
    __MAX_FIRING_RATE = 40.0
    __RIPPLE_DETECTION_TIMEOUT = 1.0
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
        plot_grid    = gridspec.GridSpec(2, 2)
        self.toolbar = NavigationToolbar(self.canvas, self.widget)
        self._rd_ax = self.figure.add_subplot(plot_grid[0])
        self._pf_ax = self.figure.add_subplot(plot_grid[1])
        self._cp_ax = self.figure.add_subplot(plot_grid[3])

        # This is tricky because right now we are using this as an array to
        # store multiple axes, each plotting a different unit.
        self._spk_pos_ax = self.figure.add_subplot(plot_grid[2])

        # Selecting individual units
        self.unit_selection = QComboBox()
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
        if clusters is None:
            self._n_total_clusters = self._spike_listener.get_n_clusters()
            # These are the clusters we are going to plot
            self._n_clusters = len(self.__CLUSTERS_TO_PLOT)
            self._clusters = self.__CLUSTERS_TO_PLOT
            self._tetrodes = self._spike_listener.get_tetrodes()
            self._n_tetrodes = len(self._tetrodes)
        else:
            # TODO: Fetch indices for these clusters
            self._n_clusters = 0
            self._clusters = None
            pass
        self._cluster_colormap = colormap.magma(np.linspace(0, 1, self._n_clusters))

        # Large arrays that are shared across processes
        self._new_ripple_frame_availale = threading.Event()
        self._shared_raw_lfp_buffer = np.reshape(np.frombuffer(ripple_buffers[0], dtype='double'), (self._n_tetrodes, RiD.LFP_BUFFER_LENGTH))
        self._shared_ripple_power_buffer = np.reshape(np.frombuffer(ripple_buffers[1], dtype='double'), (self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH))
        self._shared_place_fields = np.reshape(np.frombuffer(shared_place_fields, dtype='double'), (self._n_total_clusters, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))

        self._shared_calib_plot_means = np.reshape(np.frombuffer(calib_plot_buffers[0], dtype='double'), (RiD.CALIB_PLOT_BUFFER_LENGTH))
        self._shared_calib_plot_std_errs = np.reshape(np.frombuffer(calib_plot_buffers[1], dtype='double'), (RiD.CALIB_PLOT_BUFFER_LENGTH))

        # Local copies of the shared data that can be used at a leisurely pace
        self._lfp_tpts = np.linspace(0, RiD.LFP_BUFFER_TIME, RiD.LFP_BUFFER_LENGTH)
        self._ripple_power_tpts = np.linspace(0, RiD.LFP_BUFFER_TIME, RiD.RIPPLE_POWER_BUFFER_LENGTH)
        self._local_lfp_buffer = np.zeros((self._n_tetrodes, RiD.LFP_BUFFER_LENGTH), dtype='double')
        self._local_ripple_power_buffer = np.zeros((self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH), dtype='double')
        self._most_recent_pf = np.zeros((PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]), \
                dtype='float')

        self._spk_cnt_tpts = np.linspace(0, RiD.CALIB_PLOT_BUFFER_TIME, RiD.CALIB_PLOT_BUFFER_LENGTH)
        self._local_spk_cnt_buffer = np.zeros((RiD.CALIB_PLOT_BUFFER_LENGTH), dtype='double')
        self._local_spk_cnt_stderr_buffer = np.zeros((RiD.CALIB_PLOT_BUFFER_LENGTH), dtype='double')

        # Automatically keep only a fixed number of entries in this buffer... Useful for plotting
        self._pos_timestamps = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_x = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_y = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._speed = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)

        # Maintain a separate deque for each cluster to plot
        self._spk_pos_x = []
        self._spk_pos_y = []
        for cl_idx in range(self._n_clusters):
            self._spk_pos_x.append(deque([], self.__N_SPIKES_TO_PLOT))
            self._spk_pos_y.append(deque([], self.__N_SPIKES_TO_PLOT))

        # Figure/Animation element. So far the following have been included
        # Ripple detection
        # Place Fields
        # Position/Spikes overalaid
        self.__N_SUBPLOT_ROWS = int(np.ceil((self._n_clusters/self.__N_SUBPLOT_COLS)))
        self._rd_frame = list()
        self._spk_pos_frame = list()
        self._pf_frame = list()
        self._cp_frame = list()
        self._anim_objs = list()
        self._thread_list = list()

        # Communication buffers
        self._position_buffer = self._position_estimator.get_position_buffer_connection()
        self._spike_buffer = self._place_field_handler.get_spike_place_buffer_connection(self.__CLUSTERS_TO_PLOT)
        logging.info(MODULE_IDENTIFIER + "Graphics interface started.")
        self.setLayout()
        self.clearAxes()

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
        vbox_unit_buttons.addWidget(self.tetrode_selection)
        vbox_unit_buttons.addWidget(self.next_tet_button)
        vbox_unit_buttons.addWidget(self.prev_tet_button)
        vbox_unit_buttons.addStretch(1)
        parent_layout_box.addLayout(vbox_unit_buttons)
        QDialog.setLayout(self.widget, parent_layout_box)

    def NextUnit(self):
        QtHelperUtils.display_warning('Function not implemented!')

    def PrevUnit(self):
        QtHelperUtils.display_warning('Function not implemented!')

    def NextTetrode(self):
        QtHelperUtils.display_warning('Function not implemented!')

    def PrevTetrode(self):
        QtHelperUtils.display_warning('Function not implemented!')

    def clearAxes(self):
        # Ripple detection axis
        self._rd_ax.cla()
        self._rd_ax.set_xlabel("Time (s)")
        self._rd_ax.set_ylabel("EEG (uV)")
        self._rd_ax.set_xlim((0.0, RiD.LFP_BUFFER_TIME))
        self._rd_ax.set_ylim((-1.0, 1.0))
        self._rd_ax.grid(True)

        # Calibration plot
        self._cp_ax.cla()
        self._cp_ax.set_xlabel("Time (s)")
        self._cp_ax.set_ylabel("Spike Rate (spks/5ms)")
        self._cp_ax.set_xlim((0.0, RiD.LFP_BUFFER_TIME))
        self._cp_ax.set_ylim((0.0, 20.0))
        self._cp_ax.grid(True)

        # Place field
        self._pf_ax.cla()
        self._pf_ax.set_xlabel("x (bin)")
        self._pf_ax.set_ylabel("y (bin)")
        self._pf_ax.set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
        self._pf_ax.set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
        self._pf_ax.grid(True)

        # Spikes (from a single cell) and position
        self._spk_pos_ax.cla()
        self._spk_pos_ax.set_xlabel("x (bin)")
        self._spk_pos_ax.set_ylabel("y (bin)")
        self._spk_pos_ax.set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
        self._spk_pos_ax.set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
        self._spk_pos_ax.grid(True)

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
        if self._rd_ax:
            self._rd_frame[0].set_data(self._lfp_tpts, self._local_lfp_buffer[0,:]/self.__PEAK_LFP_AMPLITUDE)
            self._rd_frame[1].set_data(self._ripple_power_tpts, -0.5 + (self._local_ripple_power_buffer[0,:] - RippleAnalysis.D_MEAN_RIPPLE_POWER)/(2 * RippleAnalysis.D_STD_RIPPLE_POWER * RiD.RIPPLE_POWER_THRESHOLD))
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
        if self._cp_ax is not None:
            self._cp_frame[0].set_data(self._spk_cnt_tpts, self._local_spk_cnt_buffer)
            self._cp_frame[1].set_data(self._spk_cnt_tpts, self._local_spk_cnt_buffer + self._local_spk_cnt_stderr_buffer)
            self._cp_frame[2].set_data(self._spk_cnt_tpts, self._local_spk_cnt_buffer - self._local_spk_cnt_stderr_buffer)
            return self._cp_frame

    def update_position_and_spike_frame(self, step=0):
        """
        Function used for animating the current position of the animal.
        """
        if len(self._spk_pos_ax) != 0:
            # print(self._pos_x)
            # print(self._pos_y)
            # TODO: Add colors based on which cluster the spikes are coming from
            for cl_idx in range(self._n_clusters):
                self._spk_pos_frame[cl_idx].set_data((self._spk_pos_x[cl_idx], self._spk_pos_y[cl_idx]))
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
        if self._pf_ax:
            # print("Peak FR: %.2f, Mean FR: %.2f"%(np.max(self._most_recent_pf), np.mean(self._most_recent_pf)))
            # min_fr = np.min(self._most_recent_pf)
            # max_fr = np.max(self._most_recent_pf)
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
                np.copyto(self._local_lfp_buffer, self._shared_raw_lfp_buffer)
                np.copyto(self._local_ripple_power_buffer, self._shared_ripple_power_buffer)
                # print(MODULE_IDENTIFIER + "Peak ripple power in frame %.2f"%np.max(self._shared_ripple_power_buffer))
            else:
                time.sleep(self.__RIPPLE_DETECTION_TIMEOUT)
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
                np.copyto(self._local_spk_cnt_buffer, self._shared_calib_plot_means)
                np.copyto(self._local_spk_cnt_stderr_buffer, self._shared_calib_plot_std_errs)
            else:
                time.sleep(self.__RIPPLE_DETECTION_TIMEOUT)
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
            # np.copyto(self._most_recent_pf, self._shared_place_fields[self.__CLUSTERS_TO_PLOT[0], :, :])
            np.mean(self._shared_place_fields, out=self._most_recent_pf, axis=0)
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
                self._pos_timestamps.append(position_data[0])
                self._pos_x.append(position_data[1])
                self._pos_y.append(position_data[2])
                self._speed.append(position_data[3])
                logging.debug(MODULE_IDENTIFIER + "Fetched Position data... (%d, %d), v: %.2fcm/s"% \
                      (position_data[1],position_data[2], position_data[3]))
            else:
                time.sleep(self.__PLOT_REFRESH_RATE)
        logging.info(MODULE_IDENTIFIER + "Position pipe closed.")

    def process_command(self, key_in):
        user_input = self._key_entry.get()
        if user_input == 's':
            print('Pausing ripple interruption.')
            self._ripple_trigger_thread.disable()
        elif user_input == 'r':
            print('Resuming ripple interruption.')
            self._ripple_trigger_thread.enable()
        self._key_entry.delete(0, tkinter.END)
        pass
    
    def pauseAnimation(self):
        """
        Pause all animation sources.
        """
        for ao in self.anim_obj:
            ao.event_source.stop()
    
    def playAnimation(self):
        """
        Play all animation sources.
        """
        for ao in self.anim_obj:
            ao.event_source.start()

    def initialize_ripple_detection_fig(self):
        """
        Initialize figure window for showing raw LFP and ripple power.
        :returns: TODO
        """
        lfp_frame, = self._rd_ax.plot([], [], animated=True)
        ripple_power_frame, = self._rd_ax.plot([], [], animated=True)
        self._rd_frame.append(lfp_frame)
        self._rd_frame.append(ripple_power_frame)

        # Create animation object for showing the EEG
        anim_obj = animation.FuncAnimation(self.figure, self.update_ripple_detection_frame, frames=self.__N_ANIMATION_FRAMES, interval=5, blit=True)
        self._anim_objs.append(anim_obj)

    def initialize_calib_plot_fig(self):
        """
        Initialize figure window for showing raw LFP and ripple power.
        :returns: TODO
        """
        spk_cnt_frame, = self._cp_ax.plot([], [], animated=True)
        spk_cnt_plus_sterr_frame, = self._cp_ax.plot([], [], animated=True)
        spk_cnt_minus_sterr_frame, = self._cp_ax.plot([], [], animated=True)
        self._cp_frame.append(spk_cnt_frame)
        self._cp_frame.append(spk_cnt_plus_sterr_frame)
        self._cp_frame.append(spk_cnt_minus_sterr_frame)

        # Create animation object for showing the EEG
        anim_obj = animation.FuncAnimation(self.figure, self.update_calib_plot_frame, frames=self.__N_ANIMATION_FRAMES, interval=5, blit=True)
        self._anim_objs.append(anim_obj)

    def initialize_spike_pos_fig(self):
        """
        Initialize figure window for showing spikes overlaid on position
        """
        # TODO: Deal with this to only plot spikes from the unit we are interested in.
        self._pos_fig, self._spk_pos_ax = plt.subplots(self.__N_SUBPLOT_ROWS, self.__N_SUBPLOT_COLS)
        # Create graphics entries for the actual position and also each of the spike clusters
        if (self.__N_SUBPLOT_ROWS == 1) or (self.__N_SUBPLOT_COLS == 1):
            # Matplotlib returns a 1D array in this case, a 2D array otherwise
            center_frame = int(self.__N_SUBPLOT_COLS/2)
            for cl_idx in range(self._n_clusters): 
                self._spk_pos_ax[cl_idx].set_xlabel("x (bin)")
                self._spk_pos_ax[cl_idx].set_ylabel("y (bin)")
                self._spk_pos_ax[cl_idx].set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
                self._spk_pos_ax[cl_idx].set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
                self._spk_pos_ax[cl_idx].grid(True)
                spk_frame, = self._spk_pos_ax[cl_idx].plot([], [], linestyle='None', marker='o', alpha=0.4, animated=True)
                self._spk_pos_frame.append(spk_frame)
            pos_frame, = self._spk_pos_ax[center_frame].plot([], [], animated=True)
            vel_frame  = self._spk_pos_ax[center_frame].text(40.0, 2.0, 'speed = 0cm/s')
            self._spk_pos_frame.append(pos_frame)
            self._spk_pos_frame.append(vel_frame)
        else:
            for cl_idx in range(self._n_clusters): 
                grid_idx = np.unravel_index(cl_idx, (self.__N_SUBPLOT_ROWS, self.__N_SUBPLOT_COLS))
                self._spk_pos_ax[grid_idx[0]][grid_idx[1]].set_xlabel("x (bin)")
                self._spk_pos_ax[grid_idx[0]][grid_idx[1]].set_ylabel("y (bin)")
                self._spk_pos_ax[grid_idx[0]][grid_idx[1]].set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
                self._spk_pos_ax[grid_idx[0]][grid_idx[1]].set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
                self._spk_pos_ax[grid_idx[0]][grid_idx[1]].grid(True)
                spk_frame, = self._spk_pos_ax[grid_idx[0]][grid_idx[1]].plot([], [], linestyle='None', marker='o', alpha=0.4, animated=True)
                self._spk_pos_frame.append(spk_frame)
            # TODO: Change this to make it the center plot?
            pos_frame, = self._spk_pos_ax[0][0].plot([], [], animated=True)
            vel_frame  = self._spk_pos_ax[0][0].text(40.0, 2.0, 'speed = 0cm/s')
            self._spk_pos_frame.append(pos_frame)
            self._spk_pos_frame.append(vel_frame)

        anim_obj = animation.FuncAnimation(self.figure, self.update_position_and_spike_frame, frames=self.__N_ANIMATION_FRAMES, interval=5, blit=True)
        self._anim_objs.append(anim_obj)

    def initialize_place_field_fig(self):
        """
        Initialize figure window for dynamically showing place fields.
        """
        self._pf_fig = plt.figure()
        self._pf_ax = plt.axes()

        pf_heatmap = self._pf_ax.imshow(np.zeros((PositionAnalysis.N_POSITION_BINS[0], \
                PositionAnalysis.N_POSITION_BINS[1]), dtype='float'), vmin=0, \
                vmax=self.__MAX_FIRING_RATE, animated=True)
        plt.colorbar(pf_heatmap)
        self._pf_frame.append(pf_heatmap)
        anim_obj = animation.FuncAnimation(self.figure, self.update_place_field_frame, frames=self.__N_ANIMATION_FRAMES, interval=5, blit=True)
        self._anim_objs.append(anim_obj)

    def run(self):
        """
        Start a GUI, launch all the graphics windows that have been requested
        in separate threads.
        """

        self._keep_running.set()

        # Launch a thread for fetching position data constantly
        # Add options to only start some of these threads
        self._thread_list = list()
        self._thread_list.append(threading.Thread(name="PositionFetcher", daemon=True, \
                target=self.fetch_position_and_update_frames))
        self._thread_list.append(threading.Thread(name="SpikeFetcher", daemon=True, \
                target=self.fetch_spikes_and_update_frames))
        self._thread_list.append(threading.Thread(name="PlaceFieldFetched", daemon=True, \
                target=self.fetch_place_fields))
        self._thread_list.append(threading.Thread(name="RippleFrameFetcher", daemon=True, \
                target=self.fetch_incident_ripple))
        self._thread_list.append(threading.Thread(name="CalibPlotFetcher", daemon=True, \
                target=self.fetch_calibration_plot))

        for p__thread in self._thread_list:
            p__thread.start()

        # Start the animation for Spike-Position figure, place field figure
        self.initialize_ripple_detection_fig()
        self.initialize_calib_plot_fig()
        self.initialize_spike_pos_fig()
        self.initialize_place_field_fig()

        # This is a blocking command... After you exit this, everything will end.
        while self._keep_running.is_set():
            time.sleep(1.0)

        for p__thread in self._thread_list:
            p__thread.join()
        logging.info(MODULE_IDENTIFIER + "Closed GUI and display pipes")
