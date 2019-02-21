import time
import logging
from tkinter import filedialog

def getCurrentTime(self):
    return time.strftime("%H:%M:%S ")

class InterruptionLogger(object):
    """
    Logs all messages during ripple/replay disruption into a common log file.
    """

    #TODO: Make a separate thread out of the logger?
    def __init__(self, file_prefix):
        """
        Class constructor. Specify a file prefix to be used for creating logs.

        :file_prefix: Your log file is file_prefix_<DATE>_<TIME>
        """

        time_now = time.gmtime()
        self._filename = time.strftime(file_prefix + "_%Y%m%d_%H%M%S")
        logging.basicConfig(filename=self._filename, level=logging.DEBUG)
        logging.debug("Starting Log file at " + time.ctime())

    def log(self, message):
        logging.debug(getCurrentTime() + message)
        raise NotImplementedError()

    def exit(self):
        """
        Exit logging and close file
        """
        
        logging.debug("Finished logging at " + time.ctime())
        raise NotImplementedError()
