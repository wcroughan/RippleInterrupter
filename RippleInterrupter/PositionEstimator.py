# System imports
import threading

# Local imports
import SpikeAnalysis

class BayesianEstimator(threading.Thread):

    """
    Continuously keeps track of the current position of the animal
    """

    def __init__(self):
        """TODO: to be defined1. """
        threading.Thread.__init__(self)

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
            self._ripple_trigger.wait()
            # Look at the position decoding in last few ms to decide on the
            # replay. During this time, the contents of decoded position should
            # not be modified.
            # TODO

            # Reset the ripple trigger flag
            self._ripple_trigger.clear()
