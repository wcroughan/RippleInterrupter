"""
Visualization of various measures
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as colormap
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from multiprocessing import Process
import tkinter
import time
from datetime import datetime
import threading
import logging
import numpy as np

# Local Imports
import PositionAnalysis

MODULE_IDENTIFIER = "[GraphicsHandler] "

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
    __N_SPIKES_TO_PLOT = 200
    __N_ANIMATION_FRAMES = 5000
    def __init__(self, ripple_analyzer, spike_listener, position_estimator, \
            place_field_handler, ripple_trigger_condition, clusters=None):
        """TODO: to be defined1.

        :ripple_analyzer: TODO
        :spike_listener: TODO
        :position_estimator: TODO
        :place_field_handler: TODO
        :ripple_trigger_condition: TODO
        :clusters: User specified cluster indices that we should be looking at.
        """
        Process.__init__(self)

        # Graphics windows
        self._command_window = tkinter.Tk()
        tkinter.Label(self._command_window, text="Enter command to execute...").pack()
        self._key_entry = tkinter.Entry(self._command_window)
        self._key_entry.bind("<Return>", self.process_command)
        self._key_entry.pack()
        exit_button = tkinter.Button(self._command_window, text='Quit', command=self.kill_gui)
        exit_button.pack()

        self._keep_running = True
        self._ripple_analyzer = ripple_analyzer
        self._spike_listener = spike_listener
        self._position_estimator = position_estimator
        self._place_field_handler = place_field_handler
        self._ripple_trigger_condition = ripple_trigger_condition
        if clusters is None:
            self._n_clusters = self._spike_listener.get_n_clusters()
            self._clusters = range(self._n_clusters)
        else:
            # TODO: Fetch indices for these clusters
            self._n_clusters = 0
            self._clusters = None
            pass
        self._cluster_colormap = colormap.magma(np.linspace(0, 1, self._n_clusters))
        
        # Automatically keep only a fixed number of entries in this buffer... Useful for plotting
        self._pos_timestamps = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_x = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_y = deque([], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._spk_clusters = deque([], self.__N_SPIKES_TO_PLOT)
        self._spk_timestamps = deque([], self.__N_SPIKES_TO_PLOT)
        self._spk_pos_x = deque([], self.__N_SPIKES_TO_PLOT)
        self._spk_pos_y = deque([], self.__N_SPIKES_TO_PLOT)

        # Figure elements
        self._pos_fig = None
        self._spk_pos_ax = None
        self._spk_pos_frame = []

        # Communication buffers
        self._position_buffer = self._position_estimator.get_position_buffer_connection()
        self._spike_buffer = self._place_field_handler.get_spike_place_buffer_connection()
    
    def kill_gui(self):
        self._command_window.quit()
        plt.close(self._pos_fig)
        self._command_window.destroy()

    def update_position_and_spike_frame(self, step=0):
        """
        Function used for animating the current position of the animal.
        """
        if  self._spk_pos_ax is not None:
            # print(self._pos_x)
            # print(self._pos_y)
            # TODO: Add colors based on which cluster the spikes are coming from
            self._spk_pos_frame[0].set_data((self._spk_pos_x, self._spk_pos_y))
            self._spk_pos_frame[1].set_data((self._pos_x, self._pos_y))
            if step == self.__N_ANIMATION_FRAMES:
                logging.debug(MODULE_IDENTIFIER + datetime.now().strftime("Animation Finished at %H:%M:%S.%f"))
            return self._spk_pos_frame

    def fetch_spikes_and_update_frames(self):
        while self._keep_running:
            if self._spike_buffer.poll():
                spike_data = self._spike_buffer.recv()
                # TODO: Might not have to save all the spike timestamps since
                # we are already getting position data here.
                self._spk_clusters.append(spike_data[0])
                self._spk_pos_x.append(spike_data[1])
                self._spk_pos_y.append(spike_data[2])
                self._spk_timestamps.append(spike_data[3])
                logging.debug(MODULE_IDENTIFIER + "Fetched spike from cluster: %d, in bin (%d, %d). TS: %d"%spike_data)
            else:
                time.sleep(0.05)

    def fetch_position_and_update_frames(self):
        while self._keep_running:
            if self._position_buffer.poll():
                position_data = self._position_buffer.recv()
                self._pos_timestamps.append(position_data[0])
                self._pos_x.append(position_data[1])
                self._pos_y.append(position_data[2])
                logging.debug(MODULE_IDENTIFIER + "Fetched Position data... (%d, %d)"%(position_data[1],position_data[2]))
            else:
                # Wait for a while for data to appear
                time.sleep(0.05)

    def process_command(self, key_in):
        print(self._key_entry.get())
        self._key_entry.delete(0, tkinter.END)
        pass

    def run(self):
        """
        Start a GUI, launch all the graphics windows that have been requested
        in separate threads.
        """

        # Create a command window to take user inputs
        # gui_handler = threading.Thread(name="CommandWindow", daemon=True, \
        #         target=self._command_window.mainloop)

        # Launch a thread for fetching position data constantly
        # TODO: Making these threads stoppable is too much of a pain!
        position_fetcher = threading.Thread(name="PositionFetcher", daemon=True, \
                target=self.fetch_position_and_update_frames)
        spike_fetcher = threading.Thread(name="SpikeFetcher", daemon=True, \
                target=self.fetch_spikes_and_update_frames)

        position_fetcher.start()
        spike_fetcher.start()

        self._pos_fig = plt.figure()
        self._spk_pos_ax  = plt.axes()
        self._spk_pos_ax.set_xlabel("x (bin)")
        self._spk_pos_ax.set_ylabel("y (bin)")
        self._spk_pos_ax.set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
        self._spk_pos_ax.set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
        self._spk_pos_ax.grid(True)

        # Create graphics entries for the actual position and also each of the spike clusters
        pos_frame, = plt.plot([], [], animated=True)
        spk_frame, = plt.plot([], [], linestyle='None', marker='o', animated=True)
        self._spk_pos_frame.append(spk_frame)
        self._spk_pos_frame.append(pos_frame)

        anim_obj = animation.FuncAnimation(self._pos_fig, self.update_position_and_spike_frame, frames=self.__N_ANIMATION_FRAMES, interval=25, blit=True)
        plt.show()

        # This is a blocking command... After you exit this, everything will end.
        self._command_window.mainloop()
        logging.debug(MODULE_IDENTIFIER + datetime.now().strftime("Closing GUI and display pipes at %H:%M:%S.%f"))
        self._keep_running = False
        position_fetcher.join()
        spike_fetcher.join()
