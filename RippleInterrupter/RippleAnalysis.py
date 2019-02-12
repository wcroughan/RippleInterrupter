# System imports
from scipy import signal, stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import time
import serial

# Local file imports
import TrodesInterface
import RippleDefinitions as RiD

def normalizeData(in_data):
    # TODO: Might need tiling of data if there are multiple dimensions
    data_mean = np.mean(in_data, axis=0)
    data_std  = np.std(in_data, axis=0)
    norm_data = np.divide((in_data - data_mean), data_std)
    return norm_data, data_mean, data_std

def animateLFP(timestamps, lfp, ripple_power, frame_size, statistic=None):
    """
    Animate a given LFP by plotting a fixed size sliding frame.

    :timestamps: time-points for the LFP data
    :lfp: LFP (Raw/Filtered) for a single tetrode
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
        r_pow_frame.set_data(timestamps[step:step+frame_size], ripple_power[step:step+frame_size])
        lfp_measure.set_text(txt_template % timestamps[step])
        # Updating the limits is needed still so that the correct range of data
        # is displayed! It doesn't update the axis labels though - That's a
        # different ballgame!
        plot_axes.set_xlim(timestamps[step], timestamps[step+frame_size])
        return lfp_frame, r_pow_frame, lfp_measure

    def _initAnimFrame():
        # NOTE: Init function called twice! I have seen this before but still
        # don't understand why it works this way!
        # print("Initializing animation frame...")
        plot_axes.set_xlabel('Time (s)')
        plot_axes.set_ylabel('EEG (uV)')
        plot_axes.set_ylim(min(lfp), max(lfp))
        plot_axes.set_xlim(timestamps[0], timestamps[frame_size])
        return _nextAnimFrame()

    n_frames = len(timestamps) - frame_size
    lfp_anim = animation.FuncAnimation(lfp_fig, _nextAnimFrame, np.arange(0, n_frames), \
            init_func=_initAnimFrame, interval=RiD.LFP_ANIMATION_INTERVAL, \
            blit=True, repeat=False)
    plt.figure(lfp_fig.number)
    plt.show(plot_axes)

def getRippleStatistics(tetrodes, analysis_time=4):
    """
    Get ripple data statistics for a particular tetrode and a user defined time
    period.
    Added: 2019/02/19
    Archit Gupta

    :tetrodes: Indices of tetrodes that should be used for collecting the
        statistics.
    :analysis_time: Amount of time (specified in seconds) for which the data
        should be analyzed to get ripple statistics.
    :returns: Distribution of ripple power, ripple amplitude and frequency
    """

    n_tetrodes = len(tetrodes)

    # Create a ripple filter (discrete butterworth filter with cutoff
    # frequencies set at Ripple LO and HI cutoffs.)
    ripple_filter = signal.butter(RiD.LFP_FILTER_ORDER, \
            (RiD.RIPPLE_LO_FREQ, RiD.RIPPLE_HI_FREQ), \
            btype='bandpass', analog=False, output='sos', \
            fs=RiD.LFP_FREQUENCY)

    # Filter the contents of the signal frame by frame
    ripple_frame_filter = signal.sosfilt_zi(ripple_filter)

    # Tile it to take in all the tetrodes at once
    ripple_frame_filter = np.tile(np.reshape(ripple_frame_filter, \
            (RiD.LFP_FILTER_ORDER, 1, 2)), (1, n_tetrodes, 1))

    # Initialize a new client
    client = TrodesInterface.SGClient("RippleAnalyst")
    if (client.initialize() != 0):
        del client
        raise Exception("Could not initialize connection! Aborting.")

    # Access the LFP stream and create a buffer for trodes to fill LFP data into
    lfp_stream = client.subscribeLFPData(TrodesInterface.LFP_SUBSCRIPTION_ATTRIBUTE, tetrodes)
    lfp_stream.initialize()

    # LFP Sampling frequency TIMES desired analysis time period
    N_DATA_SAMPLES = analysis_time * RiD.LFP_FREQUENCY

    # Each LFP frame (I think it is just a single time point) is returned in
    # lfp_frame_buffer. The entire timeseries is stored in raw_lfp_buffer.
    lfp_frame_buffer = lfp_stream.create_numpy_array()
    raw_lfp_buffer   = np.zeros((n_tetrodes, N_DATA_SAMPLES), dtype='float')
    ripple_power     = np.zeros((n_tetrodes, N_DATA_SAMPLES), dtype='float')

    # Create a plot to look at the raw lfp data
    timestamps  = np.linspace(0, analysis_time, N_DATA_SAMPLES)
    iter_idx    = 0
    while (iter_idx < N_DATA_SAMPLES):
        n_lfp_frames = lfp_stream.available(0)
        for frame_idx in range(n_lfp_frames):
            t_stamp = lfp_stream.getData()
            raw_lfp_buffer[:, iter_idx] = lfp_frame_buffer[:]

            # If we have enough data to fill in a new filter buffer, filter the
            # new data
            if (iter_idx > 0) and (iter_idx % RiD.LFP_FILTER_ORDER == 0):
                ripple_filtered_lfp, ripple_frame_filter = signal.sosfilt(ripple_filter, \
                       raw_lfp_buffer, axis=1, zi=ripple_frame_filter)

                # TODO: Might have to average over a longer window to be able
                # to pick out ripples effectively. Ripple power is only being
                # reported for each frame right now: Filling out the same value
                # for the entire frame.
                ripple_power[:,iter_idx-RiD.LFP_FILTER_ORDER:iter_idx] = \
                        np.tile(np.reshape(np.sqrt(np.mean(np.power(ripple_filtered_lfp, 2), axis=1)), \
                        (n_tetrodes, 1)), (1, RiD.LFP_FILTER_ORDER))


            # TODO: Look at shorter time windows and see if we can isolate
            # individual ripples in the raw LFP (by eye) and in the filtered
            # data.
            """
            if ((iter_idx % 1000) == 0):
                # DEBUG: Just plot the raw LFP data frame by frame.
                wait_for_user_input = input('Press any key to continue')
            """

            iter_idx += 1
            if (iter_idx >= N_DATA_SAMPLES):
                break

    print("Collected raw LFP Data. Visualizing.")

    # Plots - Pick a tetrode to plot the data from
    # Static plots
    plt.ion()
    n_tetrodes = 1
    for tetrode_idx in range(n_tetrodes):
        plt.figure()
        plt.plot(timestamps, raw_lfp_buffer[tetrode_idx,:])
        plt.grid(True)
        plt.show()

    # Normalize both EEG and Ripple power so that they can be visualized together.
    norm_lfp, lfp_mean, lfp_std = normalizeData(raw_lfp_buffer[1,:])
    norm_ripple_power, power_mean, power_std = normalizeData(ripple_power[1,:])
    print("Ripple Statistics...")
    print("Mean: %.2f"%power_mean)
    print("Std: %.2f"%power_std)

    # Plot a histogram of the LFP power
    plt.figure()
    hist_axes = plt.axes()
    plt.hist(norm_ripple_power, bins=RiD.N_POWER_BINS, density=True)
    plt.grid(True)
    hist_axes.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.show()

    # Animation
    wait_for_user_input = input('Press ENTER to continue')
    animateLFP(timestamps, norm_lfp, norm_ripple_power, 400)

    # Program exits with a segmentation fault! Can't help this.
    wait_for_user_input = input('Press any key to continue')

if (__name__ == "__main__"):
    # tetrodes_to_be_analyzed = [1,2,3,14,15,16,17,18,19,20,21,22,23,24,25,26,32,37,39,40]
    tetrodes_to_be_analyzed = [25,14,17,18,39]
    getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed])
