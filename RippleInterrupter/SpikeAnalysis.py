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
    #               -> 
    tree_roo
    return

class PlaceField(threading.Thread):

    """
    Class for creating and updating place fields online
    """

    def __init__(self, clusters=None):
        """
        Class constructor: Initialize a thread for pooling information on this
        place field.

        :threadID: Thread ID to be attached to this Place Field
        :clusters: Spike cluster used to feed data into the place field. Fed in
            as tuples of tetrode ID and cluster ID.

        """
        threading.Thread.__init__(self)
        self._clusters = cluster

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

    def __init__(self, sg_client, tetrodes, clusters):
        """TODO: to be defined1. """
        threading.Thread.__init__(self)
        self._spike_stream = sg_client.subscribeSpikesData(TrodesInterface.SPIKE_SUBSCRIPTION_ATTRIBUTE, \
                tetrodes)
        self._spike_stream.initialize()
        self._spike_record = self._spike_stream.create_numpy_array()
        self._clusters = clusters
        self._n_clusters = 0
        for tet_clusters in self._clusters:
            self._n_clusters += len(tet_clusters)
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

