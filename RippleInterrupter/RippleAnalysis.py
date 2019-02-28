"""
Module for analysis of ripples in streaming or offline LFP data.
"""
# System imports
import os
import sys
import time
import logging
from datetime import datetime
import threading
import collections
from multiprocessing import Pipe
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Local file imports
import TrodesInterface
import ThreadExtension
import RippleDefinitions as RiD
import Visualization

MODULE_IDENTIFIER = "[RippleAnalysis] "
D_MEAN_RIPPLE_POWER = 60.0
D_STD_RIPPLE_POWER = 30.0

class RippleSynchronizer(ThreadExtension.StoppableThread):
    """
    Waits for a ripple to be detected and processes downstream changes for
    analyzing spike contents.
    """

    # Wait for 10ms while checking if the event flag is set.
    _EVENT_TIMEOUT = 0.01

    def __init__(self, sync_event):
        """TODO: to be defined1. """
        ThreadExtension.StoppableThread.__init__(self)
        self._sync_event = sync_event
        logging.debug(MODULE_IDENTIFIER + datetime.now().strftime("Started Ripple Synchronization thread at %H:%M:%S.3%f"))

    def run(self):
        while not self.req_stop():
            # TODO: EVENT TIMEOUT will act as a timeout between successive
            # ripple detections (maybe too hacky)
            logging.debug(MODULE_IDENTIFIER + "Waiting for ripple trigger...")
            with self._sync_event:
                self._sync_event.wait()
            logging.debug(MODULE_IDENTIFIER + "Ripple tiggered.")

class LFPListener(ThreadExtension.StoppableThread):
    """
    Thread that listens to the LFP stream and continuously fetches LFP timestamps and data
    """
    
    CLASS_IDENTIFIER  = "[LFPListener] "
    def __init__(self, sg_client, target_tetrodes):
        """
        Class constructor
        Subsribe to LFP stream on a given client and start listening
        to LFP data for a set of target tetrode channels.
        :sg_client: SpikeGadgets client for subscribing LFP steam
        :target_tetrodes: Set of tetrodes to listen to for ripples
        """
        ThreadExtension.StoppableThread.__init__(self)
        self._target_tetrodes = target_tetrodes
        self._n_tetrodes = len(self._target_tetrodes)
        self._lfp_producer = None

        # Data streams
        self._lfp_stream = sg_client.subscribeLFPData(TrodesInterface.LFP_SUBSCRIPTION_ATTRIBUTE, \
                self._target_tetrodes)
        self._lfp_stream.initialize()
        self._lfp_buffer = self._lfp_stream.create_numpy_array()
        logging.debug(self.CLASS_IDENTIFIER + "Started LFP listener thread.")

    def get_n_tetrodes(self):
        return self._n_tetrodes

    def get_lfp_listener_connection(self):
        self._lfp_producer, lfp_consumer = Pipe()
        return lfp_consumer

    def run(self):
        """
        Start fetching LFP frames.
        """
        while not self.req_stop():
            n_lfp_frames = self._lfp_stream.available(0)
            if n_lfp_frames == 0:
                logging.debug(self.CLASS_IDENTIFIER + "No LFP Frames to read... Sleeping.")
                time.sleep(0.005)
            for frame_idx in range(n_lfp_frames):
                timestamp = self._lfp_stream.getData()
                if self._lfp_producer is not None:
                    self._lfp_producer.send((timestamp.trodes_timestamp, self._lfp_buffer[:]))
                    logging.debug(self.CLASS_IDENTIFIER + "LFP Frame at %d sent out for ripple analysis."%timestamp.trodes_timestamp)

class RippleDetector(ThreadExtension.StoppableProcess):
    """
    Thread for ripple detection on a set of channels [ONLINE]
    """

    def __init__(self, lfp_listener, baseline_stats=None, \
            trigger_condition=None, shared_buffers=None):
        """

        :baseline_stats: Ripple power mean and std to detect/trigger interruption
        :trigger_condition: Instance of multiprocessing.Event() (or
            threading.Condition()) to communicate synchronization with other threads.
        """

        ThreadExtension.StoppableProcess.__init__(self)
        # TODO: Error handling if baseline stats are not provided - Get them by
        # looking at the data for some time.

        # Mean and standard deviation could either be provided, or estimated in
        # real time. Since the animal spends most of his time running, we can
        # probably get away by not looking at running speed to turn the
        # computation of mean and standard deviation on/off 
        self._n_tetrodes = lfp_listener.get_n_tetrodes()
        if baseline_stats is None:
            self._mean_ripple_power = np.full(self._n_tetrodes, D_MEAN_RIPPLE_POWER, dtype='float')
            self._std_ripple_power = np.full(self._n_tetrodes, D_STD_RIPPLE_POWER, dtype='float')
        else:
            self._mean_ripple_power = baseline_stats[0]
            self._std_ripple_power  = baseline_stats[1]
        self._trigger_condition = trigger_condition

        # Output connections
        self._ripple_buffer_connections = []

        # Input pipe for accessing LFP stream
        self._lfp_consumer = lfp_listener.get_lfp_listener_connection()

        # Shared variables
        self._local_lfp_buffer = collections.deque(maxlen=RiD.LFP_BUFFER_LENGTH)
        self._local_ripple_power_buffer = collections.deque(maxlen=RiD.RIPPLE_POWER_BUFFER_LENGTH)
        self._raw_lfp_buffer = np.reshape(np.frombuffer(shared_buffers[0]), (self._n_tetrodes, RiD.LFP_BUFFER_LENGTH))
        self._ripple_power_buffer = np.reshape(np.frombuffer(shared_buffers[1]), (self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH))

        # TODO: Check that initialization worked!
        self._ripple_filter = signal.butter(RiD.LFP_FILTER_ORDER, \
                (RiD.RIPPLE_LO_FREQ, RiD.RIPPLE_HI_FREQ), btype='bandpass', \
                analog=False, output='sos', fs=RiD.LFP_FREQUENCY)

        logging.debug(MODULE_IDENTIFIER + "Started Ripple detection thread.")

    def get_ripple_buffer_connections(self):
        """
        Returns a connection to the stored ripple power and raw lfp buffer
        :returns: Receiving end of the pipe for ripple buffer
        """
        my_end, your_end = Pipe()
        self._ripple_buffer_connections.append(my_end)
        return your_end

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
                (RiD.LFP_FILTER_ORDER, 1, 2)), (1, self._n_tetrodes, 1))
        # Buffers for storing/manipulating raw LFP, ripple filtered LFP and
        # ripple power.
        raw_lfp_window = np.zeros((self._n_tetrodes, RiD.LFP_FILTER_ORDER), dtype='float')
        ripple_power = collections.deque(maxlen=RiD.RIPPLE_SMOOTHING_WINDOW)
        previous_mean_ripple_power = np.zeros_like(self._mean_ripple_power)
        lfp_window_ptr = 0
        pow_window_ptr = 0
        n_data_pts_seen = 0

        # Delay measures for ripple detection (and trigger)
        prev_ripple = -np.Inf
        curr_time   = 0
        start_wall_time = time.time()
        curr_wall_time = start_wall_time
        while not self.req_stop():
            # Acquire buffered LFP frames and fill them in a filter buffer
            if self._lfp_consumer.poll():
                # print(MODULE_IDENTIFIER + "LFP Frame received for filtering.")
                (timestamp, raw_lfp_window[:, lfp_window_ptr]) = self._lfp_consumer.recv()
                self._local_lfp_buffer.append(raw_lfp_window[:, lfp_window_ptr])
                lfp_window_ptr += 1

                # If the filter window is full, filter the data and record it in rippple power
                if (lfp_window_ptr == RiD.LFP_FILTER_ORDER):
                    lfp_window_ptr = 0
                    filtered_window, ripple_frame_filter = signal.sosfilt(self._ripple_filter, \
                           raw_lfp_window, axis=1, zi=ripple_frame_filter)
                    current_ripple_power = np.sqrt(np.mean(np.power(filtered_window, 2), axis=1))
                    ripple_power.append(current_ripple_power)

                    # Fill in the shared data variables
                    self._local_ripple_power_buffer.append(current_ripple_power)

                    # TODO: Enable this part of the code to update the mean and STD over time
                    # Update the mean and std for ripple power at each of the tetrodes
                    """
                    np.copyto(previous_mean_ripple_power, self._mean_ripple_power)
                    self._mean_ripple_power += (ripple_power[:, pow_window_ptr] - previous_mean_ripple_power)/n_data_pts_seen
                    self._std_ripple_power += (ripple_power[:, pow_window_ptr] - previous_mean_ripple_power) * \
                            (ripple_power[:, pow_window_ptr] - self._mean_ripple_power)
                    # This is the accumulate sum of squares. The actual variance is <current-value>/(n_data_pts_seen-1)
                    """
                    n_data_pts_seen += 1
                    print("Read %d frames so far."%n_data_pts_seen)
        
                    # TODO: Right now, we are not using average power in the smoothing window, but the current power.
                    power_to_baseline_ratio = np.divide(current_ripple_power - self._mean_ripple_power, self._std_ripple_power)

                    # Timestamp has both trodes and system timestamps!
                    curr_time = float(timestamp)/RiD.LFP_FREQUENCY
                    logging.debug(MODULE_IDENTIFIER + "Frame @ %d filtered, mean ripple strength %.2f"%(timestamp, np.mean(power_to_baseline_ratio)))
                    if ((curr_time - prev_ripple) > RiD.RIPPLE_REFRACTORY_PERIOD):
                        # TODO: Consider switching to all, or atleast a majority of tetrodes for ripple detection.
                        if (power_to_baseline_ratio > RiD.RIPPLE_POWER_THRESHOLD).any():
                            logging.info(MODULE_IDENTIFIER + "Detected ripple at %.2f. Peak Strength: %.2f"% \
                                    (curr_wall_time-start_wall_time, np.max(power_to_baseline_ratio)))
                            prev_ripple = curr_time
                            with self._trigger_condition:
                                # First trigger interruption and all time critical operations
                                # Nothing to do right now

                                # Copy data over for visualization
                                self._raw_lfp_buffer = np.asarray(self._local_lfp_buffer)
                                self._ripple_power_buffer = np.asarray(self._local_ripple_power_buffer)
                                self._trigger_condition.notify(2)
                            curr_wall_time = time.time()
            else:
                logging.debug(MODULE_IDENTIFIER + "No LFP Frames to process. Sleeping")
                time.sleep(0.005)
"""
Code below here is from the previous iterations where we were using a single
file to detect and disrupt all ripples based on the LFP on a single tetrode.
"""
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
    tetrodes_to_be_analyzed = [24,33]
    if len(sys.argv) == 1:
        (power_mean, power_std) = getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                analysis_time=10.0)
    elif (int(sys.argv[1][0]) == 1):
        getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                ripple_statistics=[60.0, 30.0], show_ripples=True, \
                analysis_time=20)

if (__name__ == "__main__"):
    main()
