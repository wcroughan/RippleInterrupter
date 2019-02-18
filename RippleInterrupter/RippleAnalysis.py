"""
Module for analysis of ripples in streaming or offline LFP data.
"""
# System imports
import os
import sys
import time
import threading
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Local file imports
import TrodesInterface
import RippleDefinitions as RiD
import Visualization

class RippleDetector(threading.Thread):
    """
    Thread for ripple detection on a set of channels [ONLINE]
    """

    def __init__(self, sg_client, target_tetrodes):
        """
        Subsribe to LFP stream on a given client and start listening
        to/filtering data for a set of target tetrode channels.

        :sg_client: SpikeGadgets client for subscribing LFP steam
        :target_tetrodes: Set of tetrodes to listen to for ripples

        """

        threading.Thread.__init__(self)
        self._target_tetrodes = target_tetrodes
        self._n_tetrodes = len(self._target_tetrodes)
        self._lfp_stream = sg_client.subscribeLFPData(TrodesInterface.LFP_SUBSCRIPTION_ATTRIBUTE, \
                self._target_tetrodes)
        self._lfp_stream.initialize()
        # TODO: Check that initialization worked!
        self._ripple_filter = signal.butter(RiD.LFP_FILTER_ORDER, \
                (RiD.RIPPLE_LO_FREQ, RiD.RIPPLE_HI_FREQ), btype='bandpass', \
                analog=False, output='sos', fs=RiD.LFP_FREQUENCY)

        self._lfp_buffer = self._lfp_buffer.create_numpy_array()

    def run(self):
        """
        Start thread execution

        :t_max: Max amount of hardware time (measured by Trodes timestamps)
            that ripple analysis should work for.
        :returns: Nothing
        """
        # Filter the contents of the signal frame by frame
        ripple_frame_filter = signal.sosfilt_zi(self._ripple_filter)

        # Tile it to take in all the tetrodes at once
        ripple_frame_filter = np.tile(np.reshape(ripple_frame_filter, \
                (RiD.LFP_FILTER_ORDER, 1, 2)), (1, n_tetrodes, 1))
        # Buffers for storing/manipulating raw LFP, ripple filtered LFP and
        # ripple power.
        raw_lfp_window = np.zeros(self._n_tetrodes, RiD.LFP_FILTER_ORDER)
        ripple_power = np.zeros(self._n_tetrodes, RiD.RIPPLE_SMOOTHING_WINDOW)
        lfp_window_ptr = 0
        pow_window_ptr = 0
        while True:
            # Acquire buffered LFP frames and fill them in a filter buffer
            n_lfp_frames = self._lfp_stream.available(0)
            for frame_idx in range(n_lfp_frames):
                trodes_timestamp = self._lfp_stream.getData()
                raw_lfp_window[:, lfp_window_ptr] = self._lfp_buffer[:]
                lfp_window_ptr += 1

                # If the filter window is full, filter the data and record it
                # in rippple power
                if (lfp_window_ptr == RiD.LFP_FILTER_ORDER):
                    lfp_window_ptr = 0
                    filtered_window, ripple_frame_filter = signal.sosfilt(self._ripple_filter, \
                           lfp_window, axis=1, zi=ripple_frame_filter)
                    ripple_power[:, pow_window_ptr] = np.sqrt(np.mean(np.power( \
                            filtered_window, 2), axis=1))
                    pow_window_ptr += 1

                    # TODO: Add a condition to check if ripple power is at a
                    # level where we would like to check for a replay
        raise NotImplementedError()
        
def normalizeData(in_data):
    # TODO: Might need tiling of data if there are multiple dimensions
    data_mean = np.mean(in_data, axis=0)
    data_std  = np.std(in_data, axis=0)
    norm_data = np.divide((in_data - data_mean), data_std)
    return (norm_data, data_mean, data_std)

def writeLogFile(trodes_timestamps, ripple_events, wall_ripple_times, interrupt_events):
    outf = open(os.getcwd() + "/ripple_interruption_out__" +str(time.time()) + ".txt", "w")

    # First write out the ripples
    outf.write("Detected Ripple Events...\n")
    for idx, t_stamp in enumerate(trodes_timestamps):
        outf.write(str(t_stamp) + ", ")
        outf.write(str(ripple_events[idx]) + ", ")
        outf.write(str(wall_ripple_times[idx]) + ", ")
        outf.write("\n")
    outf.write("\n")

    outf.write("Interruption Events...\n")
    for i_event in interrupt_events:
        outf.write(str(i_event) + "\n")

    outf.close()

def getRippleStatistics(tetrodes, analysis_time=4, show_ripples=False, \
        ripple_statistics=None):
    """
    Get ripple data statistics for a particular tetrode and a user defined time
    period.
    Added: 2019/02/19
    Archit Gupta

    :tetrodes: Indices of tetrodes that should be used for collecting the
        statistics.
    :analysis_time: Amount of time (specified in seconds) for which the data
        should be analyzed to get ripple statistics.
    :show_ripple: Show ripple as they happen in real time.
    :ripple_statistics: Mean and STD for declaring something a sharp-wave
        ripple.
    :returns: Distribution of ripple power, ripple amplitude and frequency
    """

    if show_ripples:
        plt.ion()

    n_tetrodes = len(tetrodes)
    report_ripples = (ripple_statistics is not None)

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
    prev_interrupt = -1.0

    # Data to be logged for later use
    ripple_events = []
    trodes_timestamps = []
    wall_ripple_times = []
    interrupt_events = []
    if report_ripples:
        print('Using pre-recorded ripple statistics')
        print('Mean: %.2f'%ripple_statistics[0])
        print('Std: %.2f'%ripple_statistics[1])

    if show_ripples:
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
                        if show_ripples:
                            if (iter_idx == int((prev_ripple + RiD.INTERRUPTION_WINDOW) * RiD.LFP_FREQUENCY)):
                                data_begin_idx = int(max(0, iter_idx - 2*RiD.INTERRUPTION_TPTS))
                                interruption_axes.clear()
                                interruption_axes.plot(timestamps[data_begin_idx:iter_idx], raw_lfp_buffer[0, \
                                        data_begin_idx:iter_idx])
                                interruption_axes.scatter(prev_ripple, 0, c="r")
                                plt.grid(True)
                                plt.draw()
                                plt.pause(0.001)
                                # print(raw_lfp_buffer[0, data_begin_idx:iter_idx])

                        # If any of the tetrodes has a ripple, let's call it a ripple for now
                        ripple_to_baseline_ratio = (frame_ripple_power[0] - ripple_statistics[0])/ \
                                ripple_statistics[1]
                        if (ripple_to_baseline_ratio > RiD.RIPPLE_POWER_THRESHOLD):
                            current_time = float(iter_idx)/float(RiD.LFP_FREQUENCY)
                            if ((current_time - prev_ripple) > RiD.RIPPLE_REFRACTORY_PERIOD):
                                prev_ripple = current_time
                                current_wall_time = time.time() - start_wall_time
                                time_lag = (current_wall_time - current_time)
                                print("Ripple @ %.2f, Real Time %.2f [Lag: %.2f], strength: %.1f"%(current_time, current_wall_time, time_lag, ripple_to_baseline_ratio))
                                if interrupt_ripples and ((current_time - prev_interrupt) > RiD.INTERRUPT_REFRACTORY_PERIOD):
                                    prev_interrupt = current_time;
                                    sendBiphasicPulse(ser)
                                    # TODO: Add ripple interruption code
                                    interruption_time = time.time() - start_wall_time
                                    print("Ripple Interrupted@ %.2f!"% interruption_time)
                                    interrupt_events.append(interruption_time)
                                trodes_timestamps.append(trodes_time_stamp)
                                ripple_events.append(current_time)
                                wall_ripple_times.append(current_wall_time)

            iter_idx += 1
            if (iter_idx >= N_DATA_SAMPLES):
                break

    client.closeConnections()
    print("Collected raw LFP Data. Visualizing.")
    power_mean, power_std = Visualization.visualizeLFP(timestamps, raw_lfp_buffer, ripple_power, \
            ripple_filtered_lfp, ripple_events, do_animation=False)
    if report_ripples:
        writeLogFile(trodes_timestamps, ripple_events, wall_ripple_times, interrupt_events)

    # Program exits with a segmentation fault! Can't help this.
    wait_for_user_input = input('Press ENTER to quit')
    return (power_mean, power_std)

def main():
    # tetrodes_to_be_analyzed = [1,2,3,14,15,16,17,18,19,20,21,22,23,24,25,26,32,37,39,40]
    # tetrodes_to_be_analyzed = [23,14,17,18,39]
    tetrodes_to_be_analyzed = [3]
    if len(sys.argv) == 1:
        (power_mean, power_std) = getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                analysis_time=10.0)
    elif (int(sys.argv[1][0]) == 0):
        getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                ripple_statistics=[60.0, 30.0], show_ripples=True, \
                interrupt_ripples=True, analysis_time=20)
    elif (int(sys.argv[1][0]) == 1):
        getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                ripple_statistics=[60.0, 30.0], show_ripples=True, \
                interrupt_ripples=False, analysis_time=20)

if (__name__ == "__main__"):
    main()
