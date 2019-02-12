"""
Constant definitions that can be accessed across files without having to
respecify them at multiple locations
"""

# LFP sampling rate, and filter parameters
LFP_FREQUENCY     = 1500
LFP_FILTER_LENGTH = 0.01     # 100ms long window for filtering ripples
LFP_FILTER_ORDER  = int(LFP_FILTER_LENGTH * LFP_FREQUENCY)

# Frequency range for Ripple. Keep in mind that a high range completely gets
# rid of the sharp wave ripple - It is a little harder to see the filtered
# ripple events and tell if they were real ripples or not!
RIPPLE_LO_FREQ    = 150 
RIPPLE_HI_FREQ    = 250

# Custom constants
LFP_ANIMATION_INTERVAL = 15     # Animation Frame rate

# Plotting and aesthetics
N_POWER_BINS = 20
