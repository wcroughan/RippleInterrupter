# System imports
from scipy import signal, stats
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import serial

# Local file imports
import TrodesInterface
import RippleDefinitions as RiD

def getRippleStatistics(tetrodes, analysis_time=10):
    """
    Get ripple data statistics for a particular tetrode and a user defined time
    period.
    Added: 2019/02/19
    Archit Gupta

    :tetrodes: Indices of tetrodes that should be used for collecting the
        statistics.
    :analysis_time: Amount of time (specified in seconds) for which the data
        should be analyzed to get ripple statistics.
    :returns: Distribution of ripple power, ripple amplitude and frequency
    """

    n_tetrodes = len(tetrodes)

    # Create a ripple filter (discrete butterworth filter with cutoff
    # frequencies set at Ripple LO and HI cutoffs.)
    ripple_filter = signal.butter(RiD.LFP_FILTER_ORDER, \
            (RiD.RIPPLE_LO_FREQ, RiD.RIPPLE_HI_FREQ), \
            btype='bandpass', analog=False, output='sos', \
            fs=RiD.LFP_FREQUENCY)

    # Initialize a new client
    client = TrodesInterface.SGClient("RippleAnalyst")
    if (client.initialize() != 0):
        del client
        raise Exception("Could not initialize connection! Aborting.")

    # Access the LFP stream and create a buffer for trodes to fill LFP data into
    lfp_stream = client.subscribeLFPData(TrodesInterface.LFP_SUBSCRIPTION_ATTRIBUTE, tetrodes)
    lfp_stream.initialize()

    # TODO: Change this to account for the user specified time for analysis.
    # For now, we will just enforce a fixed number of iterations for which LFP
    # data is brought in and analyzed.
    N_DATA_SAMPLES = 1000

    # Each LFP frame (I think it is just a single time point) is returned in
    # lfp_frame_buffer. The entire timeseries is stored in raw_lfp_buffer.
    lfp_frame_buffer = lfp_stream.create_numpy_array()
    raw_lfp_buffer   = np.zeros((n_tetrodes, N_DATA_SAMPLES), dtype='float')

    # Create a plot to look at the raw lfp data
    timestamps  = np.linspace(0, analysis_time, N_DATA_SAMPLES)
    iter_idx    = 0
    while (iter_idx < N_DATA_SAMPLES):
        n_lfp_frames = lfp_stream.available(0)
        for frame_idx in range(n_lfp_frames):
            t_stamp = lfp_stream.getData()
            raw_lfp_buffer[:, iter_idx] = lfp_frame_buffer[:]

            # TODO: Look at shorter time windows and see if we can isolate
            # individual ripples in the raw LFP (by eye) and in the filtered
            # data.
            """
            if ((iter_idx % 1000) == 0):
                # DEBUG: Just plot the raw LFP data frame by frame.
                wait_for_user_input = input('Press any key to continue')
            """

            iter_idx += 1
            if (iter_idx >= N_DATA_SAMPLES):
                break

    # Plots - Pick a tetrode to plot the data from

    # Some very desparate debugging here!
    # print(raw_lfp_buffer)
    for tetrode_idx in range(n_tetrodes):
        plt.figure()
        plt.plot(timestamps, raw_lfp_buffer[tetrode_idx,:])
        plt.ion()
        plt.show()

    # Program exits with a segmentation fault! Can't help this.
    wait_for_user_input = input('Press any key to continue')

if (__name__ == "__main__"):
    tetrodes_to_be_analyzed = [1,2,3,14,15,16,17,18,19,20,21,22,23,24,25,26,32,37,39,40]
    getRippleStatistics([str(tetrode) for tetrode in tetrodes_to_be_analyzed])
