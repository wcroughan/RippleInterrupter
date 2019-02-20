#System imports
import time
import threading
from scipy import signal, stats
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from multiprocessing import Queue, Pipe
import numpy as np

# Local imports
import TrodesInterface

def readClusterFile(filename, tetrodes):
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
    raw_cluster_idx = 0
    # Some unnecessary accesses to get to tetrodes and clusters
    tree_root = cluster_tree.getroot()
    polygon_clusters = tree_root.getchildren()[0]
    ntrode_list = list(polygon_clusters)

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
    return raw_cluster_idx, n_trode_to_cluster_idx_map

class PlaceFieldHandler(threading.Thread):

    """
    Class for creating and updating place fields online
    """

    def __init__(self, position_processor, spike_processor, place_fields, place_field_lock):
    # def __init__(self, position_processor, spike_processor, place_fields):
        threading.Thread.__init__(self)
        self._position_buffer = position_processor.get_position_buffer_connection()
        self._spike_buffer = spike_processor.get_spike_buffer_connection()
        self._place_fields = place_fields
        self._nspks_in_bin = np.zeros(np.shape(place_fields))
        self._bin_occupancy = position_processor.get_bin_occupancy()
        self._has_pf_request = False
        self._place_field_lock = place_field_lock
        print(time.strftime("Started thread for building place fields at %H:%M:%S"))

    def run(self):
        """
        Start collecting data to construct the field
        :returns: Nothing
        """

        current_posbin_x = 0
        current_posbin_y = 0
        next_posbin_y = 0
        next_posbin_x = 0
        next_postime = 0
        spk_time = 0

        update_pf_every_n_spks = 100 #this controls how many spikes are collected before place fields are recalculated
        pf_update_spk_iter = 0

        while True:
            # If thread has been requested to stop an updates to place field
            # data because of an outside access to the data
            if self._has_pf_request:
                time.sleep(0.005) #5ms
                continue

            pos_buf_empty = False

            while self._spike_buffer.poll() and not self._has_pf_request:
                #note this assumes technically that spikes are in strict chronological order. Although realistically
                #we can break that assumption since that would only cause the few spikes that come late to be assigned
                #to the next place bin the animal is in

                #get the next spike
                (spk_cl, spk_time) = self._spike_buffer.recv()

                #if it's after our most recent position update, try and read the next position
                #keep reading positions until our position data is ahead of our spike data
                while not pos_buf_empty and spk_time >= next_postime:
                    current_posbin_x = next_posbin_x
                    current_posbin_y = next_posbin_y
                    if not self._position_buffer.poll():
                        #If we don't have any position data ahead of spike data,
                        #don't bother checking this every time the outer loop iterates
                        pos_buf_empty = True
                        break
                    else:
                        (next_postime, next_posbin_x, next_posbin_y) = self._position_buffer.recv()

                #add this spike to spike counts for place bin
                # print("Spike from cluster %d, in bin (%d, %d)"%(spk_cl, current_posbin_x, current_posbin_y))
                # print(current_posbin_y)
                self._nspks_in_bin[spk_cl, current_posbin_x, current_posbin_y] += 1
                pf_update_spk_iter += 1

            if pf_update_spk_iter >= update_pf_every_n_spks and not self._has_pf_request:
                pf_update_spk_iter = 0
                # Deal with divide by zero when the occupancy is zero for some of the place bins
                self._place_fields = np.divide(self._nspks_in_bin, self._bin_occupancy, \
                        out=np.zeros_like(self._nspks_in_bin), where=self._bin_occupancy!=0)






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


class SpikeDetector(threading.Thread):
    """
    Pulls spikes from Trodes and assigns them to different worker threads that
    will allocate them to place bins.
    """

    #def __init__(self, sg_client, tetrodes, spike_buffer, position_buffer):
    def __init__(self, sg_client, cluster_identity_map):
        """TODO: to be defined1. """
        threading.Thread.__init__(self)
        tetrode_argument = []
        for ntrode in cluster_identity_map:
            for cluster in cluster_identity_map[ntrode]:
                tetrode_argument.append(str(ntrode) + "," + str(cluster))
        # Take a look at all the cluster we will be listening to
        # print(tetrode_argument)
        self._spike_stream = sg_client.subscribeSpikesData(TrodesInterface.SPIKE_SUBSCRIPTION_ATTRIBUTE, \
                tetrode_argument)
        self._spike_stream.initialize()
        self._spike_record = self._spike_stream.create_numpy_array()
        self._spike_buffer_connections = []
        self._cluster_identity_map = cluster_identity_map
        print(time.strftime("Spike Detection thread started at %H:%M:%S"))
        return

    def get_spike_buffer_connection(self):
        my_end, your_end = Pipe()
        self._spike_buffer_connections.append(my_end)
        return your_end

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
                # print("Spike Timestamp %d, from Tetrode %d, Cluster %d."%(\
                #         spike_timestamp, tetrode_id, cluster_id))

                # Put this spike in the spike buffer queue
                # TODO: Can remove this if it is never a problem
                if cluster_id not in self._cluster_identity_map[tetrode_id]:
                    # Spike from an unclustered region... Ignore
                    print("Warning: Spike Ignored!")
                    continue
                unique_cluster_identity = self._cluster_identity_map[tetrode_id][cluster_id]
                # print("Spike Timestamp %d, from uClusterID %d"%(spike_timestamp,unique_cluster_identity))
                for outp in self._spike_buffer_connections:
                    outp.send((unique_cluster_identity, spike_timestamp))
