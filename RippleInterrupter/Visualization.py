"""
Visualization of various measures
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from multiprocessing import Process
import time
import threading
import numpy as np

# Local Imports
import PositionAnalysis

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
    __N_ANIMATION_FRAMES = 5000
    def __init__(self, ripple_analyzer, spike_listener, position_estimator, \
            place_field_handler, ripple_trigger_condition):
        """TODO: to be defined1.

        :spike_listener: TODO
        :position_estimator: TODO
        :place_field_handler: TODO

        """
        Process.__init__(self)
        self._ripple_analyzer = ripple_analyzer
        self._spike_listener = spike_listener
        self._position_estimator = position_estimator
        self._place_field_handler = place_field_handler
        self._ripple_trigger_condition = ripple_trigger_condition
        
        # Automatically keep only a fixed number of entries in this buffer... Useful for plotting
        self._pos_timestamps = deque(self.__N_POSITION_ELEMENTS_TO_PLOT*[0], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_x = deque(self.__N_POSITION_ELEMENTS_TO_PLOT*[0], self.__N_POSITION_ELEMENTS_TO_PLOT)
        self._pos_y = deque(self.__N_POSITION_ELEMENTS_TO_PLOT*[0], self.__N_POSITION_ELEMENTS_TO_PLOT)

        # Figure elements
        self._pos_ax = None
        self._pos_frame = None
        self._spk_ax = None
        self._spk_frame = None

        # Communication buffers
        self._position_buffer = self._position_estimator.get_position_buffer_connection()
        self._spike_buffer = self._spike_listener.get_spike_buffer_connection()

    def update_position_frame(self, step=0):
        """
        Function used for animating the current position of the animal.
        """
        if  self.pos_ax is not None:
            # print(self._pos_x)
            # print(self._pos_y)
            self._pos_frame.set_data((self._pos_x, self._pos_y))
            if step == self.__N_ANIMATION_FRAMES:
                print(time.strftime("Animation Finished at %H:%M:%S."))
            return self._pos_frame,

    def fetch_position_and_update_frames(self):
        while True:
            if self._position_buffer.poll():
                position_data = self._position_buffer.recv()
                self._pos_timestamps.append(position_data[0])
                self._pos_x.append(position_data[1])
                self._pos_y.append(position_data[2])
                print("Fetched Position data... (%d, %d)"%(position_data[1],position_data[2]))
            else:
                # Wait for a while for data to appear
                time.sleep(0.05)
        pass

    def run(self):
        """
        Start a GUI, launch all the graphics windows that have been requested
        in separate threads.
        """

        # Launch a thread for fetching position data constantly
        position_fetcher = threading.Thread(name="PositionFetcher", target=self.fetch_position_and_update_frames, \
                args=())
        position_fetcher.start()

        pos_fig = plt.figure()
        self.pos_ax  = plt.axes()
        self.pos_ax.set_xlabel("x (bin)")
        self.pos_ax.set_ylabel("y (bin)")
        self.pos_ax.set_xlim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[0]))
        self.pos_ax.set_ylim((-0.5, 0.5+PositionAnalysis.N_POSITION_BINS[1]))
        self.pos_ax.grid(True)
        self._pos_frame, = plt.plot([], [], animated=True)
        anim_obj = animation.FuncAnimation(pos_fig, self.update_position_frame, frames=self.__N_ANIMATION_FRAMES, interval=25, blit=True)
        plt.show()

        position_fetcher.join()
