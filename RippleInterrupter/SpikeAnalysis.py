#System imports
import threading
from scipy import signal, stats
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

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

        raise NotImplementedError()

class SpikeDetector(threading.Thread):
    """
    Pulls spikes from Trodes and assigns them to different worker threads that
    will allocate them to place bins.
    """

    def __init__(self, sg_client, tetrodes, spike_buffer):
        """TODO: to be defined1. """
        threading.Thread.__init__(self)
        self._spike_stream = sg_client.subscribeSpikesData(TrodesInterface.SPIKE_SUBSCRIPTION_ATTRIBUTE, \
                tetrodes)
        self._spike_stream.initialize()
        self._spike_record = self._spike_stream.create_numpy_array()
        self._spike_buffer = spike_buffer
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

                # Put this spike in the spike buffer queue
                self._spike_record.put()
