# System imports
import threading

# Local imports
import SpikeAnalysis

class BayesianEstimator(threading.Thread):
    """
    Continuously keeps track of the current position of the animal
    """

    def __init__(self, spike_buffer, place_fields):
        """TODO: to be defined1. """
        threading.Thread.__init__(self)
        # Hoping that everything in python is pass by reference. Place fields
        # is a giant array! Both spike buffer and place fields are shared
        # resources.
        self._spike_buffer = spike_buffer
        self._place_fields = place_fields

    def run(self):
        raise NotImplementedError()

class ReplayClassifier(threading.Thread):

    """
    When triggered by a ripple (or an external event)"""

    REPLAY_ANALYSIS_WINDOW = []
    def __init__(self, ripple_trigger):
        """
        Constructor. Whenever there is a ripple_trigger, look at the replay
        content and analyze if it is of interest to use (and therefore needs to
        be interrupted.)

        :ripple_trigger: Event used to start looking at a replay and deciding
            if we will interrupt it.
        """

        threading.Thread.__init__(self)
        self._ripple_trigger = ripple_trigger
        
    def run(self):
        # TODO: Need to have some condition here to make sure that threads can
        # be close off properly.
        while (True):

            # with here is used as a simple way to acquire lock on ripple_trigger
            with self._ripple_trigger:
                self._ripple_trigger.wait()
                # Look at the position decoding in last few ms to decide on the
                # replay. During this time, the contents of decoded position should
                # not be modified.
