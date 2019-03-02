#System imports
import os
import time
import threading
from datetime import datetime
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, center_of_mass
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from multiprocessing import Queue, Pipe, Condition
from tkinter import Tk, filedialog
import numpy as np
import logging

# Local imports
import RippleDefinitions as RiD
import PositionAnalysis
import TrodesInterface
import ThreadExtension

# Profiling specific code
import cProfile
MODULE_IDENTIFIER = "[SpikeAnalysis] "

def readClusterFile(filename=None, tetrodes=None):
    """
    Reads a cluster file and generates a list of tetrodes that have cells and
    all the clusters on that tetrode.

    :filename: XML file containing clustering information.
    :tetrodes: Which tetrodes to look at in the cluster file.
    :returns: A dictionary giving valid cluster indices for each tetrode.
    """
    if filename is None:
        gui_root = Tk()
        gui_root.wm_withdraw()
        filename = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select Cluster file",filetypes = (("cluster files","*.trodesClusters"),("all files","*.*")))

    try:
        cluster_tree = ET.parse(filename)
    except Exception as err:
        print(err)
        return

    # The file is organized as:
    # [ROOT] SpikeSortInfo 
    #       -> PolygonClusters
    #           -> ntrode (nTrodeID)
    #               -> cluster (clusterIndex)

    n_trode_to_cluster_idx_map = {}
    raw_cluster_idx = 0
    # Some unnecessary accesses to get to tetrodes and clusters
    tree_root = cluster_tree.getroot()
    polygon_clusters = tree_root.getchildren()[0]
    ntrode_list = list(polygon_clusters)

    if tetrodes is None:
        tetrodes = range(1,1+len(ntrode_list))

    for ti in tetrodes:
        # Offset by 1 because Trodes tetrodes start with 1!
        ntrode = ntrode_list[ti-1]
        tetrode_idx = ntrode.get('nTrodeIndex')
        if len(list(ntrode)) == 0:
            # Has no clusters on it
            continue

        # TODO: These indices go from 1.. N. Might have to switch to 0.. N if
        # that is what spike data returns.
        cluster_idx_to_id_map = {}
        for cluster in ntrode:
            local_cluster_idx = cluster.get('clusterIndex')
            cluster_idx_to_id_map[int(local_cluster_idx)] = raw_cluster_idx
            raw_cluster_idx += 1
        n_trode_to_cluster_idx_map[ti] = cluster_idx_to_id_map

    # Final value of raw_cluster_idx is a proxy for the total number of units we have
    logging.debug(MODULE_IDENTIFIER + "Cluster map... \n", n_trode_to_cluster_idx_map)
    return raw_cluster_idx, n_trode_to_cluster_idx_map

class PlaceFieldHandler(ThreadExtension.StoppableProcess):

    """
    Class for creating and updating place fields online
    """
    CLASS_IDENTIFIER = "[PlaceFieldHandler] "
    _ALLOWED_TIMESTAMPS_LAG = 10000

    def __init__(self, position_processor, spike_processor, shared_place_fields):
    # def __init__(self, position_processor, spike_processor, place_fields):
        ThreadExtension.StoppableProcess.__init__(self)
        self._position_buffer = position_processor.get_position_buffer_connection()
        self._spike_buffer = spike_processor.get_spike_buffer_connection()
        n_units = spike_processor.get_n_clusters()
        self._field_shape = (PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1])
        self._place_fields = np.reshape(np.frombuffer(shared_place_fields, dtype='double'), (n_units, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))
        self._log_place_fields = np.zeros_like(self._place_fields)
        self._nspks_in_bin = np.zeros(np.shape(self._place_fields))
        self._bin_occupancy = np.zeros((PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]), dtype='float')
        self._has_pf_request = False
        self._place_field_lock = Condition()
        self._spike_place_buffer_connections = []
        self._field_statistics_connection = None
        self._requested_clusters = []
        logging.debug(self.CLASS_IDENTIFIER + time.strftime("Started thread for building place fields at %H:%M:%S"))

    def get_field_CoM(self, cluster_idx=None):
        """
        Get the CoM for a particular place field
        """
        with self._place_field_lock:
            if cluster_idx is None:
                # TODO: Implement returning the CoMs for all the clusters here.
                raise NotImplementedError()
            else:
                return center_of_mass(self._place_fields[cluster_idx, :, :])

    def get_peak_firing_location(self, cluster_idx=None):
        """
        Get the location on the map for which the firing rate is the highest
        for a paricular field
        """
        with self._place_field_lock:
            if cluster_idx is None:
                raise NotImplementedError(0
            else:
                return np.unravel_indices(np.argmax(self._place_fields[cluster_idx, :, :]), \
                        self._field_shape)

    def get_spike_place_buffer_connection(self, cluster_idx):
        my_end, your_end = Pipe()
        self._spike_place_buffer_connections.append(my_end)
        for cl in cluster_idx:
            self._requested_clusters.append(cl)
        return your_end

    def run(self):
        """
        Start collecting data to construct the field
        :returns: Nothing
        """

        curr_postime = 0
        curr_posbin_x = 0
        curr_posbin_y = 0
        prev_posbin_y = 0
        prev_posbin_x = 0
        prev_postime = np.Inf
        last_postime = 0
        curr_speed = 0
        spk_time = 0

        update_pf_every_n_spks = 100 #this controls how many spikes are collected before place fields are recalculated
        pf_update_spk_iter = 0

        while not self.req_stop():
            # If thread has been requested to stop an updates to place field
            # data because of an outside access to the data
            if self._has_pf_request:
                logging.debug(self.CLASS_IDENTIFIER + "Waiting for Place field request to complete.")
                time.sleep(0.001) #5ms

            if not self._spike_buffer.poll():
                # logging.debug(self.CLASS_IDENTIFIER + "Spike buffer empty, sleeping")
                time.sleep(0.001)
                continue

            # BUG: If position thread is lagging, it will send the position at
            # a later time but we will keep filling the spikes at the oldest
            # position bin we ever saw. We need to wait for the position thread
            # to catch up.
            while self._spike_buffer.poll() and not self._has_pf_request:
                # logging.debug(MODULE_IDENTIFIER + "Main loop rentry.")
                #note this assumes technically that spikes are in strict chronological order. Although realistically
                #we can break that assumption since that would only cause the few spikes that come late to be assigned
                #to the next place bin the animal is in

                #get the next spike
                (spk_cl, spk_time) = self._spike_buffer.recv()
                logging.info(self.CLASS_IDENTIFIER + "Received spike from %d at %d"%(spk_cl, spk_time))

                #if it's after our most recent position update, try and read the next position
                #keep reading positions until our position data is ahead of our spike data
                while self._position_buffer.poll() and (spk_time >= curr_postime):
                    (curr_postime, curr_posbin_x, curr_posbin_y, curr_speed) = self._position_buffer.recv()

                    logging.info(self.CLASS_IDENTIFIER + "Received new position (%d, %d) at %d"%(curr_posbin_x, curr_posbin_y, curr_postime))
                    
                    # NOTE: We have to do some repeated computation here but
                    # passing occupancy from PositionAnalysis to this process
                    # was turning out to be a pain
                    timestamps_in_prev_bin = curr_postime - prev_postime
                    real_time_spent_in_prev_bin = float(timestamps_in_prev_bin)/RiD.SPIKE_SAMPLING_FREQ

                    # This should also take care of negative jumps in
                    # timestamps, leading to negative place fields.

                    # NOTE: Do not add time spent in bins when the target is
                    # stationary.. Since we are not counting spikes during this
                    # period, this penalizes bins where the animal stops.
                    if (timestamps_in_prev_bin > 0) and (curr_speed > RiD.MOVE_VELOCITY_THRESOLD):
                        logging.info(self.CLASS_IDENTIFIER + "Updating occupancy in bin (%d, %d), time spent %.2fs"%\
                                (prev_posbin_x,prev_posbin_y,real_time_spent_in_prev_bin))
                        self._bin_occupancy[prev_posbin_x, prev_posbin_y] += real_time_spent_in_prev_bin

                    prev_posbin_x = curr_posbin_x
                    prev_posbin_y = curr_posbin_y
                    prev_postime  = curr_postime

                #add this spike to spike counts for place bin
                # print("Spike from cluster %d, in bin (%d, %d)"%(spk_cl, current_posbin_x, current_posbin_y))
                # print(current_posbin_y)

                if curr_speed > RiD.MOVE_VELOCITY_THRESOLD:
                    self._nspks_in_bin[spk_cl, curr_posbin_x, curr_posbin_y] += 1
                    # Send this to the visualization pipeline to see how spike are being reported
                    if spk_cl in self._requested_clusters:
                        for pipe_in in self._spike_place_buffer_connections:
                            pipe_in.send((spk_cl, curr_posbin_x, curr_posbin_y, spk_time))
                        logging.debug(self.CLASS_IDENTIFIER + "Spike at %d sent out to listeners"%spk_time)
                else:
                    logging.info(self.CLASS_IDENTIFIER + "Spike at %d skipped, speed %.2fcm/s below threshold"%(spk_time, curr_speed))
                pf_update_spk_iter += 1

                # If spike timestamp starts leading position timestamps by too
                # much, wait for position timestamps to catch up. This
                # basically forces us to check for a new position entry after
                # each spike has gone by, minimizing incorrect reporting.
                spike_position_lag = float(spk_time) - float(curr_postime)
                if (spike_position_lag > self._ALLOWED_TIMESTAMPS_LAG):
                    logging.info(self.CLASS_IDENTIFIER + "Position lagging spikes by %d timestamps. S.%d, P.%d"%(spike_position_lag, spk_time, curr_postime))
                    curr_speed = 0
                    break

            if pf_update_spk_iter >= update_pf_every_n_spks and not self._has_pf_request:
                logging.info(MODULE_IDENTIFIER + "Updating place fields. Last spike at %d"%spk_time)
                with self._place_field_lock:
                    pf_update_spk_iter = 0
                    # Deal with divide by zero when the occupancy is zero for some of the place bins
                    # If we assign the values to a new location, new memory is allocated!
                    np.divide(self._nspks_in_bin, self._bin_occupancy, out=self._place_fields, \
                            where=self._bin_occupancy!=0)
                    # Apply gaussian smoothing to the computed place fields`
                    gaussian_filter(self._place_fields, sigma=3, output=self._place_fields)
                    np.log(self._place_fields, out=self._log_place_fields, where=self._place_fields!=0)
                    logging.info(self.CLASS_IDENTIFIER + "Peak FR: %.2f, Mean FR: %.2f"%(np.max(self._place_fields), np.mean(self._place_fields)))
                logging.debug(self.CLASS_IDENTIFIER + "Fields updated.")

    def submit_immediate_request(self):
        """
        Indicate that another thread wants to access the place field. This will
        cause the PlaceFieldHandler to immediately pause calculation and leave
        the current result, which is faster than waiting for it to finish. This
        function blocks until this pause action is complete.
        BE SURE TO CALL end_immediate_request() immediately upon finishing access to
        the place field
        """
        self._has_pf_request = True
        with self._place_field_lock:
            return

    def end_immediate_request(self):
        """
        Call this after calling submit_pf_request immediately after place field
        access is finished
        """
        self._has_pf_request = False

    def get_bin_occupancy(self):
        with self._place_field_lock:
            logging.info("Bin occupancy requested. Peak: %.2fs, Mean: %.2fs"%\
                    (np.max(self._bin_occupancy), np.mean(self._bin_occupancy)))
            return np.copy(self._bin_occupancy)

    def get_raw_place_fields(self, cluster_idx=None):
        """
        Get the raw place fields. These are to be used for visualization. For
        decoding, get log place fields instead.

        :cluster_idx: Cluster for which place field is needed. If no argument
            is provided, all fields will be returned.
        :returns: Matrix of values (N_BINS_X, N_BINS_Y) giving firing rate as a
            function of position on the field.
        """
        with self._place_field_lock:
            if cluster_idx is None:
                logging.info(MODULE_IDENTIFIER + "Raw place fields requested.")
                return np.copy(self._place_fields)
            logging.info(self.CLASS_IDENTIFIER + "Raw place fields requested for cluster %d. Peak FR: %.2f"%(cluster_idx, np.max(self._place_fields[cluster_idx, :, :])))
            return np.copy(self._place_fields[cluster_idx, :, :])

    def get_log_place_fields(self, cluster_idx=None):
        """
        Get the log place fields. These are to be used for decoding. For
        visualization, get log place fields instead.

        :cluster_idx: Cluster for which place field is needed. If no argument
            is provided, all fields will be returned.
        :returns: Matrix of values (N_BINS_X, N_BINS_Y) giving firing rate as a
            function of position on the field.
        """
        logging.debug(MODULE_IDENTIFIER + "Log place fields requested.")
        with self._place_field_lock:
            if cluster_idx is None:
                logging.info(MODULE_IDENTIFIER + "Raw place fields requested.")
                return np.copy(self._log_place_fields)
            logging.info(MODULE_IDENTIFIER + "Raw place fields requested for cluster %d"%cluster_idx)
            return np.copy(self._log_place_fields[cluster_idx, :, :])


class SpikeDetector(ThreadExtension.StoppableThread):
    """
    Pulls spikes from Trodes and assigns them to different worker threads that
    will allocate them to place bins.
    """

    #def __init__(self, sg_client, tetrodes, spike_buffer, position_buffer):
    def __init__(self, sg_client, cluster_identity_map):
        """TODO: to be defined1. """
        ThreadExtension.StoppableThread.__init__(self)
        tetrode_argument = []
        self._n_clusters = 0
        self._tetrodes = list(cluster_identity_map.keys())
        for ntrode in cluster_identity_map:
            for cluster in cluster_identity_map[ntrode]:
                tetrode_argument.append(str(ntrode) + "," + str(cluster))
                self._n_clusters += 1

        self._last_recorded_tstamp = np.zeros((self._n_clusters, 1), dtype='float')
        self._most_recent_timestamp = 0
        # Take a look at all the cluster we will be listening to
        # print(tetrode_argument)
        self._spike_stream = sg_client.subscribeSpikesData(TrodesInterface.SPIKE_SUBSCRIPTION_ATTRIBUTE, \
                tetrode_argument)
        self._spike_stream.initialize()
        self._spike_record = self._spike_stream.create_numpy_array()
        self._spike_buffer_connections = []
        self._cluster_identity_map = cluster_identity_map
        logging.debug(MODULE_IDENTIFIER + datetime.now().strftime("Spike Detection thread started at %H:%M:%S.%f"))
        return

    def get_n_clusters(self):
        return self._n_clusters

    def get_tetrodes(self):
        return self._tetrodes

    def get_spike_buffer_connection(self):
        my_end, your_end = Pipe()
        self._spike_buffer_connections.append(my_end)
        return your_end

    def run(self):
        """
        Start collecting all spikes from trodes and allocate them to differnet
        place fields!
        """
        if __debug__:
            code_profiler = cProfile.Profile()
            profile_prefix = "spike_fetcher_profile"
            profile_filename = time.strftime(profile_prefix + "_%Y%m%d_%H%M%S.pr")
            code_profiler.enable()

        while not self.req_stop():
            n_available_spikes = self._spike_stream.available(0)
            if n_available_spikes == 0:
                time.sleep(0.001)

            for spk_idx in range(n_available_spikes):
                # Populate the spike record
                timestamp = self._spike_stream.getData()

                # One entry is populated automatically in self._spike_record
                spike_timestamp = self._spike_record[0]['timestamp']
                tetrode_id = self._spike_record[0]['ntrodeid']
                cluster_id = self._spike_record[0]['cluster']

                # Get the position corresponding to this spike using the position buffer
                # print("Spike Timestamp %d, from Tetrode %d, Cluster %d."%(\
                #         spike_timestamp, tetrode_id, cluster_id))

                # Put this spike in the spike buffer queue
                """
                if cluster_id not in self._cluster_identity_map[tetrode_id]:
                    # Spike from an unclustered region... Ignore
                    logging.debug(MODULE_IDENTIFIER + "Warning: Spike Ignored!")
                    continue
                """
                unique_cluster_identity = self._cluster_identity_map[tetrode_id][cluster_id]
                if __debug__:
                    """
                    cluster_timestamp_jump = float(spike_timestamp) - self._last_recorded_tstamp[unique_cluster_identity]
                    timestamp_jump = float(spike_timestamp) - self._most_recent_timestamp
                    if cluster_timestamp_jump < 0:
                        logging.warning(MODULE_IDENTIFIER + "Backward timestamp jump in uClusterID %d, jump %d"%(unique_cluster_identity, cluster_timestamp_jump))
                    elif timestamp_jump < 0:
                        logging.warning(MODULE_IDENTIFIER + "Backward timestamp jump across uClusterIDs, jump %d"%timestamp_jump)
                    self._last_recorded_tstamp[unique_cluster_identity] = float(spike_timestamp)
                    self._most_recent_timestamp = self._last_recorded_tstamp[unique_cluster_identity]
                    """

                logging.debug(MODULE_IDENTIFIER + "Spike Timestamp %d, from uClusterID %d"%(spike_timestamp,unique_cluster_identity))
                for outp in self._spike_buffer_connections:
                    outp.send((unique_cluster_identity, spike_timestamp))
                logging.debug(MODULE_IDENTIFIER + "Spike at %d sent to listeners."%spike_timestamp)

        if __debug__:
            code_profiler.disable()
            code_profiler.dump_stats(profile_filename)
        logging.info(MODULE_IDENTIFIER + "Spike processing loop exitted!")
