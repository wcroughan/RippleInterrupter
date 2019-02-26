"""
Collect position data from trodes
"""
import threading
import time
from copy import copy
from datetime import datetime
import logging
import numpy as np
from multiprocessing import Pipe

# Local imports
import RippleDefinitions as RiD
import ThreadExtension

MODULE_IDENTIFIER = "[PositionAnalysis] "
N_POSITION_BINS = (16, 16)
FIELD_SIZE = (100, 100) # Actual size of the field in cms
class PositionEstimator(ThreadExtension.StoppableThread):
    """
    Run a thread that collects position data from trodes.
    """

    # Min/Max position values in x and y to be used for binning
    __P_MIN_X = 200
    __P_MIN_Y = -100
    __P_MAX_X = 1200
    __P_MAX_Y = 1000
    __P_BIN_SIZE_X = (__P_MAX_X - __P_MIN_X)
    __P_BIN_SIZE_Y = (__P_MAX_Y - __P_MIN_Y)
    __REAL_BIN_SIZE_X = FIELD_SIZE[0]/__P_BIN_SIZE_X
    __REAL_BIN_SIZE_Y = FIELD_SIZE[1]/__P_BIN_SIZE_Y
    __SPEED_SMOOTHING_FACTOR = 0.1

    #def __init__(self, sg_client, n_bins, past_position_buffer, camera_number=1):
    def __init__(self, sg_client, n_bins=N_POSITION_BINS, camera_number=1):
        ThreadExtension.StoppableThread.__init__(self)
        self._data_field = np.ndarray([], dtype=[('timestamp', 'u4'), ('line_segment', 'i4'), \
                ('position_on_segment', 'f8'), ('position_x', 'i2'), ('position_y', 'i2')])
        # TODO: Take the camera number into account here. This could just be
        # the index of the camera window that is open and should be connected
        # to.
        self._position_consumer = sg_client.subscribeHighFreqData("PositionData", "CameraModule")
        self._n_bins_x = int(n_bins[0])
        self._n_bins_y = int(n_bins[1])
        # self._bin_occupancy = np.zeros((self._n_bins_x, self._n_bins_y), dtype='float')
        #self._past_position_buffer = past_position_buffer

        # TODO: This is assuming that the jump in timestamps will not
        # completely fill up the memory. If the bin size is small, we might end
        # up filling the whole memory. We need this to  get appropriate
        # position bins for spikes in case the threads reading position and
        # spikes are not synchronized.
        if (self._position_consumer is None):
            # Failed to open connection to camera module
            logging.debug("Failed to open Camera Module")
            raise Exception("Could not connect to camera, aborting.")
        self._position_consumer.initialize()
        logging.debug(MODULE_IDENTIFIER + datetime.now().strftime("Starting Position tracking thread at %H:%M:%S.%f"))

        self._position_buffer_connections = []

    def getPositionBin(self):
        """
        Get the BIN for the current position.
        """
        x_bin, y_bin = self.getXYBin()
        return x_bin * self._n_bins_y + y_bin

    def getXYBin(self):
        """
        Get the x and y BIN for the current position.
        """
        px = self._data_field['position_x']
        py = self._data_field['position_y']
        x_bin = np.floor_divide(self._n_bins_x * (px - self.__P_MIN_X),self.__P_BIN_SIZE_X)
        # Camera data coming in has flipped Y-coordinates!
        y_bin = np.floor_divide(self._n_bins_y * (self.__P_MAX_Y - py),self.__P_BIN_SIZE_Y)
        return (x_bin, y_bin)

    """
    def get_bin_occupancy(self):
        return np.copy(self._bin_occupancy)
    """

    def get_position_buffer_connection(self):
        my_end, your_end = Pipe()
        self._position_buffer_connections.append(my_end)
        return your_end

    def run(self):
        """
        Collect position data continuously and update the amount of time spent
        in each position bin
        """
        # Keep track of current and previous BIN ID, and also the time at which last jump happened
        curr_x_bin = -1
        curr_y_bin = -1
        prev_x_bin = -1
        prev_y_bin = -1
        last_velocity = 0
        thread_start_time = time.time()

        # TODO: Because it will not be possible to get the correct first time
        # stamp, we will have to ignore the first data entry obtained here.
        # Otherwise it will skew the occupancy!
        prev_step_timestamp = 0
        while not self.req_stop():
            n_available_frames = self._position_consumer.available(0)
            for frame_idx in range(n_available_frames):
                self._position_consumer.readData(self._data_field)
                (curr_x_bin, curr_y_bin) = self.getXYBin()
                # DEBUG - Desparate
                """
                logging.debug(MODULE_IDENTIFIER + "Received Position data at %.2f \
                        (%d,%d)"%(time.time()-thread_start_time, \
                            self._data_field['position_x'], \
                            self._data_field['position_y']))
                """

                if (prev_x_bin < 0):
                    # First recorded bin!
                    #self._past_position_buffer.put((self._data_field['timestamp'], curr_bin_id))
                    for outp in self._position_buffer_connections:
                        outp.send((self._data_field['timestamp'], curr_x_bin, curr_y_bin, 0.0))
                    prev_x_bin = curr_x_bin
                    prev_y_bin = curr_y_bin
                    logging.debug(MODULE_IDENTIFIER + "Position Started (%d, %d)"%(curr_x_bin, curr_y_bin))
                    prev_step_timestamp = copy(self._data_field['timestamp'])
                elif ((curr_x_bin != prev_x_bin) or (curr_y_bin != prev_y_bin)):
                    current_timestamp = self._data_field['timestamp']
                    time_spent_in_prev_bin = current_timestamp - prev_step_timestamp

                    # This is some serious overkill.. Most of the times, we
                    # will be moving by just 1 position bin... That too either
                    # in X or Y
                    real_time_spent_in_prev_bin = float(time_spent_in_prev_bin)/RiD.SPIKE_SAMPLING_FREQ
                    real_distance_moved = np.sqrt(pow(curr_x_bin-prev_x_bin,2) + pow(curr_y_bin-prev_y_bin,2))
                    logging.debug(MODULE_IDENTIFIER + "Moved %.2fcm in %.2fs."%(real_distance_moved,real_time_spent_in_prev_bin))
                    if (time_spent_in_prev_bin != 0):
                        last_velocity = (1 - self.__SPEED_SMOOTHING_FACTOR) * real_distance_moved/real_time_spent_in_prev_bin + \
                                self.__SPEED_SMOOTHING_FACTOR * last_velocity

                    #self._past_position_buffer.put((prev_step_timestamp, prev_bin_id))
                    
                    for outp in self._position_buffer_connections:
                        outp.send((current_timestamp, curr_x_bin, curr_y_bin, last_velocity))
                    #self._past_position_buffer.put((current_timestamp, x_bin, y_bin))

                    # Update the total time spent in the bin we were previously in
                    # self._bin_occupancy[prev_x_bin, prev_y_bin] += real_time_spent_in_prev_bin
                    # print(np.max(self._bin_occupancy))

                    # DEBUG: Report the jump in position bins
                    logging.debug(MODULE_IDENTIFIER + "Position jumped (%d, %d) -> (%d,%d), TS:%d"%(prev_x_bin, prev_y_bin, curr_x_bin, curr_y_bin, current_timestamp))
                    # logging.debug(MODULE_IDENTIFIER + "Position binned (%d, %d) = (%d,%d)"%(curr_x_bin, curr_y_bin, \
                    #       self._data_field['position_x'], self._data_field['position_y']))

                    # Update the current bin and timestamps
                    # An assignment here just binds the variable
                    # prev_step_timestamp to current_timestamp, never giving us
                    # the actual time  jump... Mystery
                    prev_step_timestamp = copy(current_timestamp)
                    prev_x_bin = curr_x_bin
                    prev_y_bin = curr_y_bin
