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
from multiprocessing import Pipe, Lock, Event, Value
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Local file imports
import SerialPort
import Configuration
import TrodesInterface
import ThreadExtension
import RippleDefinitions as RiD
import Visualization

# Profiling
import cProfile

MODULE_IDENTIFIER = "[RippleAnalysis] "
D_MEAN_RIPPLE_POWER = 80.0
D_STD_RIPPLE_POWER = 55.0

class RippleSynchronizer(ThreadExtension.StoppableProcess):
    """
    Waits for a ripple to be detected and processes downstream changes for
    analyzing spike contents.
    """

    # Wait for 10ms while checking if the event flag is set.
    _EVENT_TIMEOUT = 1.0
    _SPIKE_BUFFER_SIZE = 200
    CLASS_IDENTIFIER = "[RippleSynchronizer] "

    def __init__(self, sync_event, spike_listener, position_estimator, place_field_handler):
        """TODO: to be defined1. """
        ThreadExtension.StoppableProcess.__init__(self)
        self._sync_event = sync_event
        self._spike_buffer = collections.deque(maxlen=self._SPIKE_BUFFER_SIZE)
        self._spike_histogram = collections.Counter()
        self._spike_buffer_connection = spike_listener.get_spike_buffer_connection()
        self._position_buffer_connection = position_estimator.get_position_buffer_connection()
        self._place_field_handler = place_field_handler
        self._position_access = Lock()
        self._spike_access = Lock()
        # TODO: This functionality should be moved to the parent class
        self._enable_synchrnoizer = Lock()
        self._is_disabled = Value("b", False)
        self._clusters_of_interest = [Configuration.EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_A[:], \
                Configuration.EXPERIMENT_DAY_20190307__INTERESTING_CLUSTERS_B[:]]
        print(self._clusters_of_interest)

        # Position data at the time ripple is triggered
        self._pos_x = -1
        self._pos_y = -1
        self._most_recent_speed = 0
        self._most_recent_pos_timestamp = 0
        self._serial_port = None
        try:
            self._serial_port = SerialPort.BiphasicPort()
        except Exception as err:
            logging.warning("Unable to open Serial port.")
            print(err)
        logging.info(self.CLASS_IDENTIFIER + "Started Ripple Synchronization thread.")

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
                            (spike_cluster in self._clusters_of_interest[1])
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
                        print(self.CLASS_IDENTIFIER + "Ripple tiggered. Loc (%d, %d), V %.2fcm/s" \
                                %(self._pos_x, self._pos_y, self._most_recent_speed))
                        logging.info(self.CLASS_IDENTIFIER + "Ripple tiggered. Loc (%d, %d), V %.2fcm/s" \
                                %(self._pos_x, self._pos_y, self._most_recent_speed))

                    with self._spike_access:
                        if len(self._spike_buffer) > 0:
                            # By default, returns 10 most frequent entries
                            most_spiking_unit = self._spike_histogram.most_common()[0][0]
                            most_recent_spike_time = self._spike_buffer[-1][1]
                            logging.info(self.CLASS_IDENTIFIER + "Most recent spike at %d"%most_recent_spike_time)
                            print(self._spike_histogram)
                        else:
                            print("Spike buffer empty!")

                # DEBUGGING: Print spike count from each of the clusters
                # print(self._place_field_handler.get_peak_firing_location(most_spiking_unit))

        logging.info(self.CLASS_IDENTIFIER + "Main process exited.")
        spike_fetcher.join()
        velocity_fetcher.join()
        logging.info(self.CLASS_IDENTIFIER + "Helper threads exited.")

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
        # Try opening a new connection
        self._lfp_stream = sg_client.subscribeLFPData(TrodesInterface.LFP_SUBSCRIPTION_ATTRIBUTE, \
                self._target_tetrodes)
        self._lfp_stream.initialize()
        self._lfp_buffer = self._lfp_stream.create_numpy_array()
        logging.info(self.CLASS_IDENTIFIER + "Started LFP listener thread.")

    def get_n_tetrodes(self):
        return self._n_tetrodes

    def get_lfp_listener_connection(self):
        self._lfp_producer, lfp_consumer = Pipe()
        return lfp_consumer

    def run(self):
        """
        Start fetching LFP frames.
        """
        if __debug__:
            code_profiler = cProfile.Profile()
            profile_prefix = "lfp_listener_profile"
            profile_filename = time.strftime(profile_prefix + "_%Y%m%d_%H%M%S.pr")
            code_profiler.enable()

        n_frames_fetched = 0
        while not self.req_stop():
            n_lfp_frames = self._lfp_stream.available(0)
            if n_lfp_frames == 0:
                # logging.debug(self.CLASS_IDENTIFIER + "No LFP Frames to read... Sleeping.")
                time.sleep(0.001)
            for frame_idx in range(n_lfp_frames):
                timestamp = self._lfp_stream.getData()
                n_frames_fetched += 1
                # print(self.CLASS_IDENTIFIER + "Fetched %d frames"%n_frames_fetched)
                if self._lfp_producer is not None:
                    self._lfp_producer.send((timestamp.trodes_timestamp, self._lfp_buffer[:]))
                    # logging.debug(self.CLASS_IDENTIFIER + "LFP Frame at %d sent out for ripple analysis."%timestamp.trodes_timestamp)

        if __debug__:
            code_profiler.disable()
            code_profiler.dump_stats(profile_filename)

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
        self._trigger_condition = trigger_condition[0]
        self._show_trigger = trigger_condition[1]

        # Output connections
        self._ripple_buffer_connections = []

        # Input pipe for accessing LFP stream
        self._lfp_consumer = lfp_listener.get_lfp_listener_connection()

        # Shared variables
        self._local_lfp_buffer = collections.deque(maxlen=RiD.LFP_BUFFER_LENGTH)
        self._local_ripple_power_buffer = collections.deque(maxlen=RiD.RIPPLE_POWER_BUFFER_LENGTH)
        self._raw_lfp_buffer = np.reshape(np.frombuffer(shared_buffers[0], dtype='double'), (self._n_tetrodes, RiD.LFP_BUFFER_LENGTH))
        self._ripple_power_buffer = np.reshape(np.frombuffer(shared_buffers[1], dtype='double'), (self._n_tetrodes, RiD.RIPPLE_POWER_BUFFER_LENGTH))

        # TODO: Check that initialization worked!
        self._ripple_filter = signal.butter(RiD.LFP_FILTER_ORDER, \
                (RiD.RIPPLE_LO_FREQ, RiD.RIPPLE_HI_FREQ), btype='bandpass', \
                analog=False, output='sos', fs=RiD.LFP_FREQUENCY)

        logging.info(MODULE_IDENTIFIER + "Started Ripple detection thread.")

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
        ripple_unseen = False
        prev_ripple = -np.Inf
        curr_time   = 0
        start_wall_time = time.time()
        curr_wall_time = start_wall_time
        while not self.req_stop():
            # Acquire buffered LFP frames and fill them in a filter buffer
            if self._lfp_consumer.poll():
                # print(MODULE_IDENTIFIER + "LFP Frame received for filtering.")
                (timestamp, current_lfp_frame) = self._lfp_consumer.recv()
                raw_lfp_window[:, lfp_window_ptr] = current_lfp_frame
                self._local_lfp_buffer.append(current_lfp_frame)
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
                    # print("Read %d frames so far."%n_data_pts_seen)
        
                    # TODO: Right now, we are not using average power in the smoothing window, but the current power.
                    power_to_baseline_ratio = np.divide(current_ripple_power - self._mean_ripple_power, self._std_ripple_power)

                    # Timestamp has both trodes and system timestamps!
                    curr_time = float(timestamp)/RiD.SPIKE_SAMPLING_FREQ
                    logging.debug(MODULE_IDENTIFIER + "Frame @ %d filtered, mean ripple strength %.2f"%(timestamp, np.mean(power_to_baseline_ratio)))
                    if ((curr_time - prev_ripple) > RiD.RIPPLE_REFRACTORY_PERIOD):
                        # TODO: Consider switching to all, or atleast a majority of tetrodes for ripple detection.
                        if (power_to_baseline_ratio > RiD.RIPPLE_POWER_THRESHOLD).any():
                            logging.info(MODULE_IDENTIFIER + "Detected ripple at %.2f. Peak Strength: %.2f"% \
                                    (curr_time, np.max(power_to_baseline_ratio)))
                            prev_ripple = curr_time
                            with self._trigger_condition:
                                # First trigger interruption and all time critical operations
                                self._trigger_condition.notify()
                                curr_wall_time = time.time()
                                ripple_unseen = True
                    if ((curr_time - prev_ripple) > RiD.LFP_BUFFER_TIME/2) and ripple_unseen:
                        ripple_unseen = False
                        # Copy data over for visualization
                        if len(self._local_lfp_buffer) == RiD.LFP_BUFFER_LENGTH:
                            np.copyto(self._raw_lfp_buffer, np.asarray(self._local_lfp_buffer).T)
                            np.copyto(self._ripple_power_buffer, np.asarray(self._local_ripple_power_buffer).T)
                            logging.info(MODULE_IDENTIFIER + "%.2fs: Peak ripple power in frame %.2f"%(curr_time, np.max(self._ripple_power_buffer)))
                            with self._show_trigger:
                                # First trigger interruption and all time critical operations
                                self._show_trigger.notify()
            else:
                # logging.debug(MODULE_IDENTIFIER + "No LFP Frames to process. Sleeping")
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
                analysis_time=100.0)
    elif (int(sys.argv[1][0]) == 1):
        getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed], \
                ripple_statistics=[60.0, 30.0], show_ripples=True, \
                analysis_time=100.0)

if (__name__ == "__main__"):
    main()
