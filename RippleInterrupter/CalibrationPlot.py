from multiprocessing import Pipe, Condition
import numpy as np
import ThreadExtension
import RippleDefinitions as RiD

class CalibrationPlot(ThreadExtension.StoppableProcess):

    """
    A basic class designed especially for gathering helpful information. Judging kilowatt levels' merit necessitates obtaining pictures quantifying relative spike times. Upon viewing, w x y z
    """
    CLASS_IDENTIFIER = "[CalibrationPlot] "

    def __init__(self, sg_client, shared_buffers, spike_processor):
    # def __init__(self, position_processor, spike_processor, place_fields):
        ThreadExtension.StoppableProcess.__init__(self)
        self._spike_buffer = spike_processor.get_spike_buffer_connection()
        self._sg_client = sg_client
        self._win_width = RiD.CALIB_PLOT_WINDOW_LENGTH * RiD.SPIKE_SAMPLING_FREQ
        self._num_spk_bins = 300
        self._bin_times = np.zeros((self._num_spk_bins))
        self._spike_count_online = np.zeros((self._num_spk_bins))
        self.max_spk_iter = 100
        self.num_bins_plot_each_side = RiD.CALIB_PLOT_BUFFER_LENGTH / 2
        self._spike_counts = np.zeros((0,RiD.CALIB_PLOT_BUFFER_LENGTH))
    
        self._shared_calib_plot_means = np.reshape(np.frombuffer(shared_buffers[0], dtype='double'), (RiD.CALIB_PLOT_BUFFER_LENGTH))
        self._shared_calib_plot_std_errs = np.reshape(np.frombuffer(shared_buffers[1], dtype='double'), (RiD.CALIB_PLOT_BUFFER_LENGTH))
        self._buffer_lock = Condition()


    def run(self):
        while not self.req_stop():
            if not self._spike_buffer.poll():
                # logging.debug(self.CLASS_IDENTIFIER + "Spike buffer empty, sleeping")
                time.sleep(0.001)
                continue

            with self._buffer_lock:
                self.spk_iter = 0
            
                while self._spike_buffer.poll() and self.spk_iter < self.max_spk_iter:
                    #get the next spike
                    (spk_cl, spk_time) = self._spike_buffer.recv()
                    logging.debug(self.CLASS_IDENTIFIER + "Received spike from %d at %d"%(spk_cl, spk_time))
                    bin = (spk_time // self._win_width) % self._num_spk_bins

                    if spk_time > self._bin_times[bin] + self._win_width:
                        new_time = self._win_width * (spk_time // self._win_width)
                        this_bin = bin
                        while self._bin_times[this_bin] < new_time:
                            self._bin_times[this_bin] = new_time
                            self._spike_count_online[this_bin] = 0

                            this_bin -= 1
                            if this_bin == -1:
                                this_bin = self._num_spk_bins - 1
                            new_time -= self._win_width

                            how_far_back = this_bin

                        self._latest_bin = bin

                    self._spike_count_online[bin] += 1
                    self.spk_iter += 1


    def mark_ripple(self):
        ripple_trodes_ts = self._sg_client.latestTrodesTimestamp()
        self.ripple_bin = (ripple_trodes_ts // self._win_width) % self._num_spk_bins


    def update_shared_buffer(self):
        with self._buffer_lock:
            new_spks = np.zeros((1,self.num_bins_plot_each_side*2))
            b1 = self.ripple_bin - self.num_bins_plot_each_side
            b2 = self.ripple_bin + self.num_bins_plot_each_side
            if b1 < 0:
                new_spks[0:(- b1)] = self._spike_count_online[(b1 + self._num_spk_bins):]
                new_spks[(-b1):self.num_bins_plot_each_side] = self._spike_count_online[0:(self.num_bins_plot_each_side+b1)]
                new_spks[self.num_bins_plot_each_side:] = self._spike_count_online[(self.num_bins_plot_each_side+b1):b2]
            elif b2 > self._num_spk_bins:
                new_spks[0:self.num_bins_plot_each_side] = self._spike_count_online[b1:(b1+self.num_bins_plot_each_side)]
                new_spks[self.num_bins_plot_each_side:(self._num_spk_bins - self.ripple_bin+self.num_bins_plot_each_side)] = self._spike_count_online[self.ripple_bin:]
                new_spks[(self._num_spk_bins - self.ripple_bin+self.num_bins_plot_each_side):] = self._spike_count_online[0:(self.num_bins_plot_each_side-self._num_spk_bins+self.ripple_bin)]
            else:
                new_spks[0:] = self._spike_count_online[b1:b2]

            self._spike_counts = np.vstack((self._spike_counts, new_spks))
            means = np.mean(self._spike_counts)
            std_errs = np.divide(np.std(self._spike_counts), np.sqrt(self._spike_counts.shape[0]))

            np.copyto(self._shared_calib_plot_means, means)
            np.copyto(self._shared_calib_plot_std_errs, std_errs)


