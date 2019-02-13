# System imports
from scipy import signal, stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
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

def sendBiphasicPulse(serial_port):
    serial_port.setDTR(True)
    time.sleep(0.0001)
    serial_port.setDTR(False)
    time.sleep(0.0001)
    serial_port.setRTS(True)
    time.sleep(0.0001)
    serial_port.setRTS(False)
    time.sleep(0.001)
    return

def getRippleStatistics(tetrodes, analysis_time=4, interruption_statistics=None, \
        show_interruptions=False, interrupt_ripples=False):
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
    BAUDRATE     = 9600
    DEFAULT_PORT = '/dev/ttyS0'

    outf = open(os.getcwd() + "/ripple_interruption_out__" +str(time.time()) + ".txt", "w")
    ser = None
    if interrupt_ripples:
        ser = serial.Serial(DEFAULT_PORT, BAUDRATE, timeout=0, stopbits=serial.STOPBITS_ONE, \
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE)

    if show_interruptions:
        plt.ion()

    n_tetrodes = len(tetrodes)
    report_ripples = (interruption_statistics is not None)

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
    N_DATA_SAMPLES = int(analysis_time * RiD.LFP_FREQUENCY)

    # Each LFP frame (I think it is just a single time point) is returned in
    # lfp_frame_buffer. The entire timeseries is stored in raw_lfp_buffer.
    lfp_frame_buffer = lfp_stream.create_numpy_array()
    ripple_filtered_lfp = np.zeros((n_tetrodes, N_DATA_SAMPLES), dtype='float')
    raw_lfp_buffer   = np.zeros((n_tetrodes, N_DATA_SAMPLES), dtype='float')
    ripple_power     = np.zeros((n_tetrodes, N_DATA_SAMPLES), dtype='float')

    # Create a plot to look at the raw lfp data
    timestamps  = np.linspace(0, analysis_time, N_DATA_SAMPLES)
    iter_idx    = 0
    prev_ripple = -1.0

    if report_ripples:
        ripple_events = []
        print('Using pre-recorded ripple statistics')
        print('Mean: %.2f'%interruption_statistics[0])
        print('Std: %.2f'%interruption_statistics[1])

    if show_interruptions:
        interruption_fig = plt.figure()
        interruption_axes = plt.axes()
        plt.plot([], [])
        plt.grid(True)
        plt.ion()
        plt.show()

    wait_for_user_input = input("Press Enter to start!")
    start_time  = 0.0
    start_wall_time = time.time()
    interruption_iter = -1
    is_first_ripple = True
    while (iter_idx < N_DATA_SAMPLES):
        n_lfp_frames = lfp_stream.available(0)
        for frame_idx in range(n_lfp_frames):
            # print("t__%.2f"%(float(iter_idx)/float(RiD.LFP_FREQUENCY)))
            t_stamp = lfp_stream.getData()
            trodes_time_stamp = client.latestTrodesTimestamp()
            raw_lfp_buffer[:, iter_idx] = lfp_frame_buffer[:]

            # If we have enough data to fill in a new filter buffer, filter the
            # new data
            if (iter_idx > RiD.RIPPLE_SMOOTHING_WINDOW) and (iter_idx % RiD.LFP_FILTER_ORDER == 0):
                lfp_frame = raw_lfp_buffer[:,iter_idx-RiD.LFP_FILTER_ORDER:iter_idx]
                # print(lfp_frame)
                filtered_frame, ripple_frame_filter = signal.sosfilt(ripple_filter, \
                       lfp_frame, axis=1, zi=ripple_frame_filter)
                # print(filtered_frame)
                ripple_filtered_lfp[:,iter_idx-RiD.LFP_FILTER_ORDER:iter_idx] = filtered_frame

                # Averaging over a longer window to be able to pick out ripples effectively.
                # TODO: Ripple power is only being reported for each frame
                # right now: Filling out the same value for the entire frame.
                frame_ripple_power = np.sqrt(np.mean(np.power( \
                        ripple_filtered_lfp[:,iter_idx-RiD.RIPPLE_SMOOTHING_WINDOW:iter_idx], 2), axis=1))
                ripple_power[:,iter_idx-RiD.LFP_FILTER_ORDER:iter_idx] = \
                        np.tile(np.reshape(frame_ripple_power, (n_tetrodes, 1)), (1, RiD.LFP_FILTER_ORDER))
                if report_ripples:
                    if is_first_ripple:
                        is_first_ripple = False
                    else:
                        # Show the previous interruption after a sufficient time has elapsed
                        if show_interruptions:
                            if (iter_idx == int((prev_ripple + RiD.INTERRUPTION_WINDOW) * RiD.LFP_FREQUENCY)):
                                data_begin_idx = int(max(0, iter_idx - 2*RiD.INTERRUPTION_TPTS))
                                interruption_axes.clear()
                                interruption_axes.plot(timestamps[data_begin_idx:iter_idx], raw_lfp_buffer[0, \
                                        data_begin_idx:iter_idx])
                                interruption_axes.scatter(prev_ripple, 0, c="r")
                                plt.draw()
                                plt.pause(0.001)
                                # print(raw_lfp_buffer[0, data_begin_idx:iter_idx])

                        # If any of the tetrodes has a ripple, let's call it a ripple for now
                        ripple_to_baseline_ratio = (frame_ripple_power[0] - interruption_statistics[0])/ \
                                interruption_statistics[1]
                        if (ripple_to_baseline_ratio > RiD.RIPPLE_POWER_THRESHOLD):
                            current_time = float(iter_idx)/float(RiD.LFP_FREQUENCY)
                            if ((current_time - prev_ripple) > RiD.RIPPLE_REFRACTORY_PERIOD):
                                ripple_events.append(current_time)
                                prev_ripple = current_time
                                current_wall_time = time.time() - start_wall_time
                                print("Ripple @ %.2f, Real Time %.2f, strength: %.1f"%(current_time, current_wall_time, ripple_to_baseline_ratio))
                                if interrupt_ripples:
                                    sendBiphasicPulse(ser)
                                    # TODO: Add ripple interruption code
                                    interruption_time = time.time() - start_wall_time
                                    print("Ripple Interrupted@ %.2f!"% interruption_time)
                                outf.write(str(trodes_time_stamp) + ", ")
                                outf.write(str(current_time) + ", ")
                                outf.write(str(current_wall_time))
                                outf.write("\n")

            iter_idx += 1
            if (iter_idx >= N_DATA_SAMPLES):
                break

    outf.close()
    print("Collected raw LFP Data. Visualizing.")
    visualizeLFP(timestamps, raw_lfp_buffer, ripple_power, ripple_filtered_lfp, \
            ripple_events, do_animation=False)

    # Program exits with a segmentation fault! Can't help this.
    wait_for_user_input = input('Press ENTER to quit')

def main():
    # tetrodes_to_be_analyzed = [1,2,3,14,15,16,17,18,19,20,21,22,23,24,25,26,32,37,39,40]
    # tetrodes_to_be_analyzed = [23,14,17,18,39]
    tetrodes_to_be_analyzed = [3]
    """
    getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
            interruption_statistics=[60.0, 40.0], show_interruptions=True, \
            interrupt_ripples=False, analysis_time=20)
    """
    if (int(sys.argv[1][0]) == 0):
        while (True):
                getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                            interruption_statistics=[60.0, 30.0], show_interruptions=False, \
                            interrupt_ripples=True, analysis_time=20)
    elif (int(sys.argv[1][0]) == 1):
                getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                            interruption_statistics=[60.0, 30.0], show_interruptions=False, \
                            interrupt_ripples=False, analysis_time=20)
    else:
        getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
            analysis_time=10.0)

if (__name__ == "__main__"):
    main()
