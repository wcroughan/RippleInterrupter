# System imports
import threading
import multiprocessing
import numpy as np

# Local imports
import RippleDefinitions
import SpikeAnalysis
import PositionAnalysis
import StimHardware
import ThreadExtension

# Size of the decoding window and the window size
# Cut 1: We are only focusing on behavioral time-scale decoding. There will not
# be a sliding window. We will just look at the posterior across disjoint
# decoding windows.
DECODING_TIME_WINDOW = 0.050
DECODING_WINDOW_SLIDE = 0.050
POSTERIOR_BUFFER_SIZE = 100

class BayesianEstimator(ThreadExtension.StoppableProcess):
    """
    When a ripple trigger arrives, switches to replay decoding immediately.
    """

    def __init__(self, spike_sender, place_field_provider, shared_place_fields):
        """TODO: to be defined1. """
        ThreadExtension.StoppableProcess.__init__(self)
        # Hoping that everything in python is pass by reference. Place fields
        # is a giant array! Both spike buffer and place fields are shared
        # resources.
        self.prob_buffer_size = POSTERIOR_BUFFER_SIZE
        self.time_bin_width = int(DECODING_TIME_WINDOW * RippleDefinitions.SPIKE_SAMPLING_FREQ) #trodes samples at 30kHz, so this is 10ms
        self.nspks_until_get_new_pf = 50
        self.num_return_time_bins = 10

        self._spike_buffer = spike_sender.get_spike_buffer_connection()
        #self._log_place_fields = place_field_provider.get_log_place_fields()
        self._place_field_provider = place_field_provider

        n_units = spike_sender.get_n_clusters()
        self._shared_place_fields = np.reshape(np.frombuffer(shared_place_fields, dtype='double'), \
            (n_units, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))
        self._most_recent_pf = np.zeros((n_units, PositionAnalysis.N_POSITION_BINS[0], \
            PositionAnalysis.N_POSITION_BINS[1]), dtype='float')

        # TODO: Might have to move to a fixed sized deque at a later point.
        # Right now, it seems, as soon as the posterior buffer is full, we
        # erase everything and start writing stuff from the first available
        # array entry.
        self._bin_times = np.zeros((1,self.prob_buffer_size))
        self.time_bin = 0
        
        self._request_made = False
        self._log_probs_out = np.zeros((self.prob_buffer_size, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))
        self._probs_out = np.add(np.zeros_like(self._log_probs_out), 1.0 / (PositionAnalysis.N_POSITION_BINS[0] + PositionAnalysis.N_POSITION_BINS[1]))
        self._output_lock = multiprocessing.Condition()


    def run(self):
        spk_iter = self.nspks_until_get_new_pf + 1

        while not self.req_stop():
            if spk_iter >= self.nspks_until_get_new_pf:
                self._place_field_provider.submit_immediate_request()
                np.log(self._shared_place_fields, out=self._most_recent_pf)
                self._place_field_provider.end_immediate_request()
                spk_iter = 0
            
            how_far_back = -1

            # Not running this for now... Too much to DEBUG
            return

            with self._output_lock:
                while (not self._request_made) and self._spike_buffer.poll():
                    #get the next spike
                    (spk_cl, spk_time) = self._spike_buffer.recv()

                    self.time_bin = (spk_time // self.time_bin_width) % self.prob_buffer_size

                    # If the timestamp of the new spike indicates that we need
                    # to change the current time-bin for which we are
                    # calculating the posterior.
                    if spk_time > self._bin_times[self.time_bin] + self.time_bin_width:
                        #update bin times
                        new_time = self.time_bin_width * (spk_time // self.time_bin_width)
                        this_bin = self.time_bin
                        while self._bin_times[this_bin] < new_time:
                            self._bin_times[this_bin] = new_time
                            self._log_probs_out[this_bin,:,:] = np.zeros_like(self._log_probs_out[this_bin,:,:])
                            self._probs_out[this_bin,:,:] = self._correction_factor
                        
                            this_bin -= 1
                            if this_bin == -1:
                                this_bin = self.prob_buffer_size - 1
                            new_time -= self.time_bin_width

                            how_far_back = this_bin



                    self._log_probs_out[self.time_bin,:,:] += self._log_place_fields[spk_cl,:,:]
                    spk_iter += 1

                # TODO: This bit of the code might have to be re-written.
                if how_far_back != -1:
                    upd_bin = how_far_back
                    if upd_bin == this_bin:
                        upd_bin += 1
                        if upd_bin == self.prob_buffer_size:
                            upd_bin = 0

                    while upd_bin != self.time_bin:
                        tmp = self._probs_out[upd_bin,:,:] 
                        tmp += self._log_probs_out[upd_bin,:,:]
                        tmp = tmp - np.mean(tmp, axis=(1,2))
                        tmp = np.exp(tmp)
                        sum = np.sum(tmp)
                        while np.isinf(sum):
                            tmp /= 2.0
                            sum = np.sum(tmp)

                        self._probs_out[upd_bin,:,:] = np.divide(tmp, sum)
                
                        upd_bin += 1
                        if upd_bin == self.prob_buffer_size:
                            upd_bin = 0




    def get_recent_probs(self):
        self._request_made = True
        with self._output_lock:
            if self.time_bin >= self.num_return_time_bins:
                ret = np.copy(self._probs_out[(self.time_bin-self.num_return_time_bins):self.time_bin,:,:])
                self._request_made = False
                return self._probs_out[(self.time_bin-self.num_return_time_bins):self.time_bin,:,:]
        
            #gotta loop around
            ret = np.concatenate((self._probs_out[(self.prob_buffer_size-self.num_return_time_bins+self.time_bin):self.prob_buffer_size,:,:], self._probs_out[0:time_bin,:,:]), axis=0)
            self._request_made = False
            return ret
 
    def get_num_returned_time_bins(self):
        return self.num_return_time_bins


class ReplayClassifier(threading.Thread):

    """
    When triggered by a ripple (or an external event)"""

    REPLAY_ANALYSIS_WINDOW = []
    def __init__(self, ripple_trigger, replay_decoder):
        """
        Constructor. Whenever there is a ripple_trigger, look at the replay
        content and analyze if it is of interest to use (and therefore needs to
        be interrupted.)

        :ripple_trigger: Event used to start looking at a replay and deciding
            if we will interrupt it.
        """

        threading.Thread.__init__(self)
        self._ripple_trigger = ripple_trigger
        self._replay_decoder = bayesianDecoder
        self._serial_port = StimHardware.SerialPort()

        self._zero_probs = np.zeros( (self._replay_decoder.get_num_returned_time_bins(), PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))
        self._condition_top = [[[False for i in range(PositionAnalysis.N_POSITION_BINS[1])] for j in range(PositionAnalysis.N_POSITION_BINS[0])] for k in range(self._replay_decoder.get_num_returned_time_bins())]
        self._condition_bottom = [[[False for i in range(PositionAnalysis.N_POSITION_BINS[1])] for j in range(PositionAnalysis.N_POSITION_BINS[0])] for k in range(self._replay_decoder.get_num_returned_time_bins())]

        if PositionAnalysis.N_POSITION_BINS[0] != 6 or PositionAnalysis.N_POSITION_BINS[1] != 6:
            raise AssertionError()

        self._condition_bottom[:,5,2:6] = True
        self._condition_top[:,0,0:4] = True

        self.min_probs_for_stim = np.sum(np.where(self._condition_top, np.ones_like(self._zero_probs), self._zero_probs)) / np.sum(np.ones_like(self._zero_probs))
        
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

                probs = self._replay_decoder.get_recent_probs()

                #TODO what are our criteria for stimulation?
                top_sum = np.sum(np.where(self._condition_top, probs, self._zero_probs))
                bot_sum = np.sum(np.where(self._condition_bottom, probs, self._zero_probs))
                should_stim = top_sum > self.min_probs_for_stim and top_sum > bot_sum

                if should_stim:
                    self._serial_port.sendBiphasicPulse()
                    print(time.strftime("Stimulated at %H:%M:%S"))


class BehaviorDecoder(threading.Thread):
    """
    Does a rough conversion from replay-timescale decoded posterior to behavior
    timescale. Note this isn't as accuracte as it could be if we had a whole 
    thread that kept track of its own posterior, but it might give us some idea
    of decoding accuracy
    """

    def __init__(self, bayesianDecoder):
        self._replay_decoder = bayesianDecoder

    def get_behavior_post(self):
        probs = self._replay_decoder.get_recent_probs()
        collapsed_probs = np.prod(probs, axis=1)
        return np.divide(collapsed_probs, np.sum(collapsed_probs))
