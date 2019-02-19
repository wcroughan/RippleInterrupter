#System imports
import time
import threading
from scipy import signal, stats
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from multiprocessing import Queue
import numpy as np

# Local imports
import TrodesInterface

def readClusterFile(filename):
    """
    Reads a cluster file and generates a list of tetrodes that have cells and
    all the clusters on that tetrode.

    :filename: XML file containing clustering information.
    :returns: A dictionary giving valid cluster indices for each tetrode.
    """

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
    # Some unnecessary accesses to get to tetrodes and clusters
    tree_root = cluster_tree.getroot()
    polygon_clusters = tree_root.getchildren()[0]
    for ntrode in polygon_clusters:
        tetrode_idx = ntrode.get('nTrodeIndex')
        if len(list(ntrode) == 0):
            # Has no clusters on it
            continue

        # TODO: These indices go from 1.. N. Might have to switch to 0.. N if
        # that is what spike data returns.
        cluster_idx_to_id_map = {}
        n_tet_clusters = 0
        for raw_idx, cluster in enumerate(ntrode):
            cluster_idx_to_id_map[int(cluster.get('clusterIndex'))] = raw_idx
            n_tet_clusters += 1
        n_trode_to_cluster_idx_map[tetrode_idx] = cluster_idx_to_id_map
    return

class PlaceFieldHandler(threading.Thread):

    """
    Class for creating and updating place fields online
    """

    def __init__(self, clusters, field_container):
        """
        Class constructor: Initialize a thread for pooling information on this
        place field.

        :threadID: Thread ID to be attached to this Place Field
        :clusters: Spike cluster used to feed data into the place field. Fed in
            as tuples of tetrode ID and cluster ID.
        :field_container: 
        """
        threading.Thread.__init__(self, past_position_buffer)
        self._past_position_buffer = past_position_buffer

    def run(self):
        """
        Start collecting data to construct the field
        :returns: Nothing
        """

        current_posbin = 0
        next_posbin = 0
        next_postime = 0

        while True:
            if self._has_pf_request:
                time.sleep(0.005) #5ms
                continue

            while not self._spike_buffer.empty() and not self._has_pf_request:
                (spk_cl, spk_time) = self._spike_buffer.pop()
                while spk_time >= next_postime:
                    current_posbin = next_posbin
                    if self._past_position_buffer.empty():
                        next_postime = np.Inf
                    else:
                        #get next time stamp and position
                        pass


    def submit_pf_request(self):
        """
        Indicate that another thread wants to access the place field. This will
        cause the PlaceFieldHandler to immediately pause calculation and leave
        the current result, which is faster than waiting for it to finish. This
        function blocks until this pause action is complete.
        BE SURE TO CALL end_pf_request() immediately upon finishing access to
        the place field
        """
        self._has_pf_request = True
        with self._place_field_lock:
            return

    def end_pf_request(self):
        """
        Call this after calling submit_pf_request immediately after place field
        access is finished
        """
        self._has_pf_request = False


            

class SpikeDetector(threading.Thread):
    """
    Pulls spikes from Trodes and assigns them to different worker threads that
    will allocate them to place bins.
    """

    def __init__(self, sg_client, tetrodes, spike_buffer, position_buffer):
        """TODO: to be defined1. """
        threading.Thread.__init__(self)
        tetrode_argument = [ tet_str + ",0" for tet_str in  tetrodes]
        self._spike_stream = sg_client.subscribeSpikesData(TrodesInterface.SPIKE_SUBSCRIPTION_ATTRIBUTE, \
                tetrode_argument)
        self._spike_stream.initialize()
        self._spike_record = self._spike_stream.create_numpy_array()
        self._spike_buffer = spike_buffer
        self._position_buffer = position_buffer
        print(time.strftime("Spike Detection thread started at %H:%M:%S"))
        return

    def run(self):
        """
        Start collecting all spikes from trodes and allocate them to differnet
        place fields!
        """
        while True:
            n_available_spikes = self._spike_stream.available(0)
            for spk_idx in range(n_available_spikes):
                # Populate the spike record
                timestamp = self._spike_stream.getData()

                # One entry is populated automatically in self._spike_record
                spike_timestamp = self._spike_record[0]['timestamp']
                tetrode_id = self._spike_record[0]['ntrodeid']
                cluster_id = self._spike_record[0]['cluster']

                # Get the position corresponding to this spike using the position buffer
                oldest_position = self._position_buffer.get()
                print("Spike Timestamp %d, position bin %d at timestamp %d matched"%(\
                        spike_timestamp, oldest_position[1], oldest_position[0]))

                # Put this spike in the spike buffer queue
                # TODO: Use unique cluster indices (or unit indices here)
                # instead of whatever has been hacked in at the moment.
                self._spike_buffer.put((tetrode_id*8 + cluster_id, spike_timestamp))
