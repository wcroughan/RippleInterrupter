# System imports
import time
import threading
import multiprocessing
import numpy as np
import collections
import logging

# Local imports
import RippleDefinitions
import SpikeAnalysis
import PositionAnalysis
import StimHardware
import ThreadExtension

# Debugging plots

# Size of the decoding window and the window size
# Cut 1: We are only focusing on behavioral time-scale decoding. There will not
# be a sliding window. We will just look at the posterior across disjoint
# decoding windows.

MODULE_IDENTIFIER = "[BayesianEstimator] "
N_FRAMES_TO_UPDATE = 4
DECODING_TIME_WINDOW = 0.025
POSTERIOR_BUFFER_SIZE = 25
POSTERIOR_SMOOTHING_FACTOR = 0.8
MIN_PLACE_CELL_FR = 1.0
MAX_PLACE_CELL_FR = 20.0

class BayesianEstimator(ThreadExtension.StoppableProcess):
    """
    Continuously decode incoming spikes at Replay time scale and send out this
    decoded data for visualization periodically.
    """

    def __init__(self, spike_sender, place_field_provider, shared_place_fields,\
            shared_posterior_buffer, trigger_condition):
        ThreadExtension.StoppableProcess.__init__(self)
        self.time_bin_width = DECODING_TIME_WINDOW * RippleDefinitions.SPIKE_SAMPLING_FREQ
        self._spike_buffer = spike_sender.get_spike_buffer_connection()
        #self._log_place_fields = place_field_provider.get_log_place_fields()
        self._place_field_provider = place_field_provider

        n_units = spike_sender.get_n_clusters()
        # Shared copy of the posterior buffer that we copy the data to when all the computation is done.
        self._shared_posterior_buffer = np.reshape(np.frombuffer(shared_posterior_buffer, dtype='double'), \
                (POSTERIOR_BUFFER_SIZE, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))

        # self._shared_place_fields = shared_place_fields
        self._shared_place_fields = np.reshape(np.frombuffer(shared_place_fields, dtype='double'), \
            (n_units, PositionAnalysis.N_POSITION_BINS[0], PositionAnalysis.N_POSITION_BINS[1]))
        self._most_recent_pf = np.zeros((n_units, PositionAnalysis.N_POSITION_BINS[0], \
            PositionAnalysis.N_POSITION_BINS[1]), dtype='float')

        # Allocate space for the Place-Field multiplier. This is the quantity
        # which we observe if we get 0 spikes from a cell.
        self._cluster_to_use = None
        self._pf_multiplier = np.ones((PositionAnalysis.N_POSITION_BINS[0], \
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
        self._place_field_provider.submit_immediate_request()
        np.copyto(self._most_recent_pf, self._shared_place_fields)
        self._place_field_provider.end_immediate_request()

        # Decide which cluster we will be using based on the firing rates of cells
        unit_peak_firing_rate = np.max(np.max(self._most_recent_pf, axis=2), axis=1)
        print(unit_peak_firing_rate)
        self._cluster_to_use = list()
        for unit_idx in range(unit_peak_firing_rate.shape[0]):
            if MIN_PLACE_CELL_FR < unit_peak_firing_rate[unit_idx] < MAX_PLACE_CELL_FR:
                self._cluster_to_use.append(unit_idx)

        self._cluster_to_use.sort()
        print("Using following clusters for decoding...")
        print(self._cluster_to_use)

        # print(self._most_recent_pf)
        # print(self._pf_multiplier)

        # Update the place field exponent - We are using all the units for now.
        # Remove the contributions in bins that have no firing from any cell..
        # This is a pain to deal with but it otherwise gets a very high
        # probability.
        np.exp(-DECODING_TIME_WINDOW*np.sum(self._most_recent_pf[self._cluster_to_use,:,:], axis=0), \
                out=self._pf_multiplier, where=np.max(self._most_recent_pf[self._cluster_to_use,:,:], axis=0)>SpikeAnalysis.EPSILON)
        # np.exp(-DECODING_TIME_WINDOW*np.sum(self._most_recent_pf, axis=0), \
        #         out=self._pf_multiplier)

        # Normalize the place field multiplier
        self._pf_multiplier = self._pf_multiplier / np.sum(self._pf_multiplier)

        # Multiply the decoding window to the place fields
        np.multiply(DECODING_TIME_WINDOW, self._most_recent_pf, out=self._most_recent_pf)

        # After every N_FRAMES_TO_UPDATE, decoded data is sent out.
        wall_time_start = time.perf_counter()
        down_time = 0.0
        elapsed_frames = 0.0
        while not self.req_stop():
            if self._spike_buffer.poll():
                down_time = 0.0
                # Get the next spike
                (spk_cl, spk_time) = self._spike_buffer.recv()
                current_time = time.perf_counter() - wall_time_start
                if spk_cl not in self._cluster_to_use:
                    continue
                logging.info(MODULE_IDENTIFIER + "Spike received CL:%d [TS]:%d. Frame (%d, %d). Time %.2f"%\
                        (spk_cl, spk_time, self._frame_start, self._frame_start + self.time_bin_width * POSTERIOR_BUFFER_SIZE,\
                        current_time))
                elapsed_frames += 1

                # If the program has just started, reset the frame start time
                if self._frame_start < 0:
                    self._time_bin = 0
                    self._frame_start = float(spk_time)
                    print(MODULE_IDENTIFIER + "First spike received [TS]:%d"%spk_time)
                    for f_idx in range(POSTERIOR_BUFFER_SIZE):
                        self._bin_times.append(self._frame_start + self.time_bin_width * f_idx)
                        self._probs_out.append(np.copy(self._pf_multiplier))
                else:
                    self._time_bin = int(np.floor_divide(float(spk_time) - self._frame_start, self.time_bin_width))
                    # logging.debug(MODULE_IDENTIFIER + "Inserting spike in time bin: %d"%self._time_bin)
                    if self._time_bin < 0:
                        # Received spike precedes the first time bin we are decoding.
                        # TODO: We probably need to increase the buffer size, or raise a warning.
                        logging.info(MODULE_IDENTIFIER + "Received spikes for cleared time bin [%d]"%self._time_bin)
                        print(MODULE_IDENTIFIER + "Received spikes for cleared time bin.")
                    elif self._time_bin >= POSTERIOR_BUFFER_SIZE:
                        # Calculate the required number of empty frames.
                        last_decoded_frame = np.copy(self._probs_out[-1])
                        n_frames_to_attach = min(POSTERIOR_BUFFER_SIZE, 1 + self._time_bin - POSTERIOR_BUFFER_SIZE)
                        elapsed_frames += n_frames_to_attach
                        self._time_bin = POSTERIOR_BUFFER_SIZE-1
                        self._frame_start += n_frames_to_attach * self.time_bin_width
                        for f_idx in range(n_frames_to_attach):
                            # Instead of initializing with the actual exponent,
                            # what if we initialize this with the last decoded
                            # frame. This might ensure continuity!
                            self._probs_out.append(np.copy(self._pf_multiplier))

                            # Doesn't seem to work very well
                            # self._probs_out.append(np.add(self._pf_multiplier, \
                            #         POSTERIOR_SMOOTHING_FACTOR * last_decoded_frame))

                            self._bin_times.append(self._frame_start + self.time_bin_width * f_idx)

                # Now that we have the correct time bin, multiply the place field in the correct place 
                try:
                    spike_field_contribution = np.multiply(self._probs_out[self._time_bin], self._most_recent_pf[spk_cl,:,:])
                    np.copyto(self._probs_out[self._time_bin], spike_field_contribution/np.sum(spike_field_contribution))

                    # Uncomment to spread spikes to neighboring time bins as well
                    if self._time_bin > 1:
                        # Add spike to the previous time bin(s) as well
                        spike_field_contribution = np.multiply(self._probs_out[self._time_bin-1], self._most_recent_pf[spk_cl,:,:])
                        np.copyto(self._probs_out[self._time_bin-1], spike_field_contribution/np.sum(spike_field_contribution))

                        # spike_field_contribution = np.multiply(self._probs_out[self._time_bin-2], self._most_recent_pf[spk_cl,:,:])
                        # np.copyto(self._probs_out[self._time_bin-2], spike_field_contribution/np.sum(spike_field_contribution))

                    if self._time_bin < POSTERIOR_BUFFER_SIZE - 1:
                        # Add spike to the next time bin as well
                        spike_field_contribution = np.multiply(self._probs_out[self._time_bin+1], self._most_recent_pf[spk_cl,:,:])
                        np.copyto(self._probs_out[self._time_bin+1], spike_field_contribution/np.sum(spike_field_contribution))

                        # spike_field_contribution = np.multiply(self._probs_out[self._time_bin+2], self._most_recent_pf[spk_cl,:,:])
                        # np.copyto(self._probs_out[self._time_bin+2], spike_field_contribution/np.sum(spike_field_contribution))
                except IndexError as err:
                    print(MODULE_IDENTIFIER + "Incorrectly accessed posterior matrix at %d"%self._time_bin)

                if elapsed_frames >= N_FRAMES_TO_UPDATE:
                    elapsed_frames = 0
                    with self._trigger:
                        # Normalize all the posterior
                        np.copyto(self._shared_posterior_buffer, np.asarray(self._probs_out))
                        # print(MODULE_IDENTIFIER + "Peak posterior %.2f in Frame 0."%np.max(self._probs_out[0]))
                        self._trigger.notify()

                    # Might as well take a break from the program and check if
                    # the application has been terminated.
                    if self.req_stop():
                        break
            else:
                # No more spikes to decode in the buffer
                down_time += 0.001
                time.sleep(0.001)
                if down_time > 10.0:
                    down_time = 0.0
                    print(MODULE_IDENTIFIER + "Warning: Not receiving spike data.")


    def fetch_decoded_estimate(self):
        with self._output_lock:
            # TODO: Copy over the value into the shared variable(s) so that the
            # decoder can be stopped externally and probed for estimate.
            blahblah = True
