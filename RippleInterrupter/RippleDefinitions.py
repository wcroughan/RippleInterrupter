"""
Constant definitions that can be accessed across files without having to
respecify them at multiple locations
"""

# Spike sampling rate
SPIKE_SAMPLING_FREQ = 30000.0

# LFP sampling rate, and filter parameters
LFP_FREQUENCY     = 1500.0
LFP_FILTER_LENGTH = 0.002     # 1ms long window for filtering ripples
LFP_FILTER_ORDER  = int(LFP_FILTER_LENGTH * LFP_FREQUENCY)
LFP_BUFFER_LENGTH = int(LFP_FREQUENCY * 1.0) # 1.0s long display window

# Frequency range for Ripple. Keep in mind that a high range completely gets
# rid of the sharp wave ripple - It is a little harder to see the filtered
# ripple events and tell if they were real ripples or not!
RIPPLE_LO_FREQ    = 150 
RIPPLE_HI_FREQ    = 250

# Refractory period for a ripple (in seconds)
RIPPLE_SMOOTHING_WINDOW  = int(0.01 * LFP_FREQUENCY) # 10ms smoothing window
RIPPLE_REFRACTORY_PERIOD = 0.2
INTERRUPT_REFRACTORY_PERIOD = 1.0

# Custom constants
LFP_ANIMATION_INTERVAL = 5     # Animation Frame rate (this is the delay between frames)
RIPPLE_POWER_THRESHOLD = 5     # Multiplicative factor for calling something a ripple
MOVE_VELOCITY_THRESOLD = 5

# Plotting and aesthetics
N_POWER_BINS = 20
INTERRUPTION_WINDOW = 0.25    # Time before/after interruption displayed
INTERRUPTION_TPTS   = int(INTERRUPTION_WINDOW * LFP_FREQUENCY)
