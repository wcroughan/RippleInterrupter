from multiprocessing import Pipe, Condition
import numpy as np
import ThreadExtension

class CalibrationPlot(ThreadExtension.StoppableProcess):

    """
    A basic class designed especially for gathering helpful information. Judging kilowatt levels' merit necessitates obtaining pictures quantifying relative spike times. Upon viewing, w x y z
    """
    CLASS_IDENTIFIER = "[CalibrationPlot] "

    def __init__(self, sg_client, spike_processor):
    # def __init__(self, position_processor, spike_processor, place_fields):
        ThreadExtension.StoppableProcess.__init__(self)
        self._spike_buffer = spike_processor.get_spike_buffer_connection()
        self._sg_client = sg_client
        #TODO subscribe to ripple events, run mark_ripple whenever they happen
        self._win_width = 0.005 * 30000.0
        self._num_spk_bins = 300
        self._bin_times = np.zeros((self._num_spk_bins))
        self._spike_count_online = np.zeros((self._num_spk_bins))
        self.max_spk_iter = 100
        self.num_bins_plot_each_side = 100
        self._spike_counts = np.zeros((0,self._num_spk_bins*2))
        self.got_ripple = False
        self.ripple_millis = 0


    def run(self):
        while not self.req_stop():
            if not self._spike_buffer.poll():
                # logging.debug(self.CLASS_IDENTIFIER + "Spike buffer empty, sleeping")
                time.sleep(0.001)
                continue

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

            if self.got_ripple and int(round(time.time() * 1000)) > self.ripple_millis + self.num_bins_plot_each_side * (self._win_width / 30):
                self.plot_spikes()
                self.got_ripple = False
                


    def mark_ripple(self):
        self.ripple_millis = int(round(time.time() * 1000))
        ripple_trodes_ts = self._sg_client.latestTrodesTimestamp()
        self.ripple_bin = (ripple_trodes_ts // self._win_width) % self._num_spk_bins
        self.got_ripple = True


    def plot_spikes(self):
        new_spks = np.zeros(1,self.num_bins_plot_each_side*2)
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
        #TODO send data to some graphics function


class CalibrationPlotTrigger(threading.Thread):
    
    """
    Listens for ripples and communicates with CalibrationPlot object defined above
    """

    REPLAY_ANALYSIS_WINDOW = []
    def __init__(self, ripple_trigger, calibplot):
        """
        Constructor. Whenever there is a ripple_trigger, look at the replay
        content and analyze if it is of interest to use (and therefore needs to
        be interrupted.)

        :ripple_trigger: Event used to start looking at a replay and deciding
            if we will interrupt it.
        """

        threading.Thread.__init__(self)
        self._ripple_trigger = ripple_trigger
        self._calibration_plot = calibplot

    def run(self):
        # TODO: Need to have some condition here to make sure that threads can
        # be close off properly.
        while (True):

            # with here is used as a simple way to acquire lock on ripple_trigger
            with self._ripple_trigger:
                self._ripple_trigger.wait()
                self._calibration_plot.mark_ripple()