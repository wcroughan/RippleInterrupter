#System imports
import threading
from scipy import signal, stats
import matplotlib.pyplot as plt

# Local imports
import TrodesInterface

class PlaceField(threading.Thread):

    """
    Class for creating and updating place fields online
    """

    def __init__(self, threadID, cluster=None):
        """
        Class constructor: Initialize a thread for pooling information on this
        place field.

        :threadID: Thread ID to be attached to this Place Field
        :cluster: Spike cluster used to feed data into the place field

        """
        threading.Thread.__init__(self)
        self._threadID = threadID
        self._cluster = cluster

    def run(self):
        """
        Start collecting data to construct the field
        :returns: Nothing
        """

        raise NotImplementedError()

class SpikeDetector(threading.Thread):

    """
    Pulls spikes from Trodes and assigns them to different worker threads that will allocate them to place bins.
    """

    def __init__(self, sg_client, tetrodes):
        """TODO: to be defined1. """
        threading.Thread.__init__(self)
        self._spike_stream = sg_client.subscribeSpikesData(TrodesInterface.SPIKE_SUBSCRIPTION_ATTRIBUTE, \
                tetrodes)
        return
