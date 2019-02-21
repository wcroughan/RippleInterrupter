"""
Extending Thread class to have a stopping event which we can set to stop the
thread and safely get all the data that the thread has at the moment.
"""

import threading

class StoppableThread(threading.Thread):

    """
    A thread which can be stopped by calling STOP function on it
    """

    def __init__(self):
        """TODO: to be defined1. """
        threading.Thread.__init__(self, daemon=True)
        self._stop_event = threading.Event()

    def stop(self):
        """
        Allows setting the _stop_event flag that can be checked to stop thread
        execution
        """
        self._stop_event.set()

    def req_stop(self):
        return self._stop_event.is_set()

    def join(self, timeout=None):
        # For all implementations of this class, need to extend the join method
        # to log the time at which thread ended.
        self.stop()
        threading.Thread.join(self, timeout)
