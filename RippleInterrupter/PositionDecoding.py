# System imports
import threading
import multiprocessing
import numpy as np
import collections

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

MODULE_IDENTIFIER = "[BayesianEstimator] "
N_FRAMES_TO_UPDATE = 20
DECODING_TIME_WINDOW = 0.050
DECODING_WINDOW_SLIDE = 0.050
POSTERIOR_BUFFER_SIZE = 100

class BayesianEstimator(ThreadExtension.StoppableProcess):
    """
    Continuously decode incoming spikes at Replay time scale and send out this
    decoded data for visualization periodically.
    """

    def __init__(self, spike_sender, place_field_provider, shared_place_fields,\
            shared_posterior_buffer, trigger_condition):
        ThreadExtension.StoppableProcess.__init__(self)
        # Hoping that everything in python is pass by reference. Place fields
        # is a giant array! Both spike buffer and place fields are shared
        # resources.
        self.time_bin_width = int(DECODING_TIME_WINDOW * RippleDefinitions.SPIKE_SAMPLING_FREQ)
        self._spike_buffer = spike_sender.get_spike_buffer_connection()
        #self._log_place_fields = place_field_provider.get_log_place_fields()
        self._place_field_provider = place_field_provider

        n_units = spike_sender.get_n_clusters()
        # Shared copy of the posterior buffer that we copy the data to when all the computation is done.
        self._shared_posterior_buffer = np.reshape(np.frombuffer(shared_posterior_buffer, dtype='double'), \
                (n_units, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))

        self._shared_place_fields = np.reshape(np.frombuffer(shared_place_fields, dtype='double'), \
            (n_units, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))
        self._most_recent_pf = np.zeros((n_units, PositionAnalysis.N_POSITION_BINS[0], \
            PositionAnalysis.N_POSITION_BINS[1]), dtype='float')

        # Allocate space for the Place-Field multiplier. This is the quantity
        # which we observe if we get 0 spikes from a cell.
        self._pf_multiplier = np.ones((n_units, PositionAnalysis.N_POSITION_BINS[0], \
            PositionAnalysis.N_POSITION_BINS[1]), dtype='float')

        # Create fixed size deque of time bins that we are currently decoding.
        # Keep track of the start time for this buffer. If spikes are received
        # before the frame start, a warning will be raised. If we get spikes
        # after the end of the time bins, a new entry will be added.
        self._bin_times = collections.deque(maxlen=POSTERIOR_BUFFER_SIZE)
        self._frame_start = -1
        self._time_bin = 0
        
        self._request_made = False
        self._probs_out = collections.deque(maxlen=POSTERIOR_BUFFER_SIZE)
        self._output_lock = multiprocessing.Condition()
        self._trigger = trigger_condition

    def run(self):
        # TODO: Not sure if we should continuously update place fields. For
        # now, we will stick to fixed place fields.

        # If we want to Calculate the log place fields and store them here.
        # np.log(self._shared_place_fields, out=self._most_recent_pf)

        # If we want to use the place fields themselves multiply them
        # (instead of adding log fields) for posterior estimation.
        np.copyto(self._most_recent_pf, self._self._shared_place_fields)

        # Update the place field exponent - We are using all the units for now.
        # TODO: Remove units from this calculation that have very low firing rates.
        np.exp(-DECODING_TIME_WINDOW*np.sum(self._most_recent_pf, axis=0), \
                out=self._pf_multiplier)

        self._place_field_provider.end_immediate_request()

        # After every N_FRAMES_TO_UPDATE, decoded data is sent out.
        elapsed_frames = 0
        while not self.req_stop():
            if self._spike_buffer.poll():
                # Get the next spike
                (spk_cl, spk_time) = self._spike_buffer.recv()

                # If the program has just started, reset the frame start time
                if self._frame_start < 0:
                    elapsed_frames = 1
                    self._time_bin = 0
                    self._frame_start = spk_time
                    self._bin_times.append(self._frame_start)
                    self._probs_out.append(self._pf_multiplier)
                else:
                    self._time_bin = (spk_time - self._frame_start) // self.time_bin_width
                    if self._time_bin < 0:
                        # Received spike precedes the first time bin we are decoding.
                        # TODO: We probably need to increase the buffer size, or raise a warning.
                        logging.info(MODULE_IDENTIFIER + "Received spikes for cleared time bin.")
                    elif self._time_bin >= POSTERIOR_BUFFER_SIZE:
                        # Calculate the required number of empty frames.
                        elapsed_frames += 1
                        n_frames_to_attach = self._time_bin - POSTERIOR_BUFFER_SIZE
                        self._time_bin -= n_frames_to_attach
                        self._frame_start += n_frames_to_attach * self.time_bin_width
                        for f_idx in range(n_frames_to_attach):
                            self._probs_out.append(self._pf_multiplier)
                            self._bin_times.append(self._frame_start + self.time_bin_width * f_idx)

                    # Now that we have the correct time bin, multiply the place field in the correct place 
                    np.multiply(self._probs_out[self._time_bin], self._most_recent_pf[spk_cl,:,:], out=self._probs_out[self._time_bin])

                    if elapsed_frames >= N_FRAMES_TO_UPDATE:
                        elapsed_frames = 0
                        with self._output_lock:
                            np.copyto(self._shared_posterior_buffer, np.asarray(self._probs_out).T)
                else:
                    # No more spikes to decode in the buffer
                    time.sleep(0.01)


    def fetch_decoded_estimate(self):
        with self._output_lock:
            # TODO: Copy over the value into the shared variable(s) so that the
            # decoder can be stopped externally and probed for estimate.
            blahblah = True
