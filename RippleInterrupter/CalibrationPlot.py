from multiprocessing import Pipe, Condition
import time
import numpy as np
import ThreadExtension
import RippleDefinitions as RiD
import logging

class CalibrationPlot(ThreadExtension.StoppableProcess):

    """
    A basic class designed especially for gathering helpful information. Judging kilowatt levels' merit necessitates obtaining pictures quantifying relative spike times. Upon viewing, w x y z
    """
    CLASS_IDENTIFIER = "[CalibrationPlot] "

    def __init__(self, sg_client, shared_buffers, spike_processor, buffer_lock):
    # def __init__(self, position_processor, spike_processor, place_fields):
        ThreadExtension.StoppableProcess.__init__(self)
        self._spike_buffer = spike_processor.get_spike_buffer_connection()
        self._sg_client = sg_client
        self._win_width = RiD.CALIB_PLOT_WINDOW_LENGTH * RiD.SPIKE_SAMPLING_FREQ
        RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE = 300
        self._bin_times = np.zeros((RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE))
        self._spike_count_online = np.zeros((RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE))
        self.max_spk_iter = 100
        self._spike_counts = np.zeros((0,RiD.CALIB_PLOT_BUFFER_LENGTH))
    
        self._shared_calib_plot_means = np.reshape(np.frombuffer(shared_buffers[0], dtype='double'), (RiD.CALIB_PLOT_BUFFER_LENGTH))
        self._shared_calib_plot_std_errs = np.reshape(np.frombuffer(shared_buffers[1], dtype='double'), (RiD.CALIB_PLOT_BUFFER_LENGTH))
        self._spike_count_online = np.reshape(np.frombuffer(shared_buffers[2], dtype='uint32'), (RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE))
        self._buffer_lock = buffer_lock


    def run(self):
        while not self.req_stop():
            if not self._spike_buffer.poll():
                logging.debug(self.CLASS_IDENTIFIER + "Spike buffer empty, sleeping")
                time.sleep(0.1)
                continue

            with self._buffer_lock:
                self.spk_iter = 0

                while self._spike_buffer.poll() and self.spk_iter < self.max_spk_iter:
                    #get the next spike
                    (spk_cl, spk_time) = self._spike_buffer.recv()
                    logging.debug(self.CLASS_IDENTIFIER + "Received spike from %d at %d"%(spk_cl, spk_time))
                    m_bin = int((spk_time // self._win_width) % RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE)

                    if spk_time > self._bin_times[m_bin] + self._win_width:
                        new_time = self._win_width * (spk_time // self._win_width)
                        this_bin = m_bin
                        while self._bin_times[this_bin] < new_time:
                            self._bin_times[this_bin] = new_time
                            self._spike_count_online[this_bin] = 0

                            this_bin -= 1
                            if this_bin == -1:
                                this_bin = RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE - 1
                            new_time -= self._win_width

                        self._latest_bin = m_bin

                    self._spike_count_online[m_bin] += 1
                    self.spk_iter += 1
        logging.info(self.CLASS_IDENTIFIER + "Spike calibration plot Stopped.")

    def update_shared_buffer(self, trodes_ts):
        with self._buffer_lock:
            new_spks = np.zeros((RiD.CALIB_PLOT_BUFFER_LENGTH))
            b2 = int((trodes_ts // self._win_width) % RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE)
            b1 = int(b2 - RiD.CALIB_PLOT_BUFFER_LENGTH)

            if b1 < 0:
                bb = -b1
                new_spks[0:(-b1)] = self._spike_count_online[(b1+RiD.CALIB_PLOT_ONLINE_BUFFER_SIZE):]
                new_spks[(-b1):] = self._spike_count_online[0:b2]
            else:
                new_spks[0:] = self._spike_count_online[b1:b2]

            self._spike_counts = np.vstack((self._spike_counts, new_spks))
            means = np.mean(self._spike_counts, axis=0)
            std_errs = np.divide(np.std(self._spike_counts, axis=0), np.sqrt(self._spike_counts.shape[0]))

            np.copyto(self._shared_calib_plot_means, means)
            np.copyto(self._shared_calib_plot_std_errs, std_errs)


