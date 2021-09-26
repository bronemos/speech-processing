"""
ELEC-E5500 Speech Processing -- Autumn 2019 Python Exercise 2:
Fundamental frequency estimation functions.
"""
import numpy as np


def fundf_autocorr(frame, fs, f0_min, f0_max, vuv_threshold):
    # EX2_FUNDF_AUTOCORR Fundamental frequency estimation with the
    # autocorrelation method.
    # Inputs:   'frame': Windowed signal segment
    #           'fs': sampling frequency
    #           'f0_min' and 'f0_max': Given in Hz, represent the search space
    #           for the F0 values
    #           'vuv_threshold': Heuristic value that acts as a classifier
    #           between voiced and unvoiced frames. A frame is classified as
    #           unvoiced if the strength of the
    #           autocorrelation peak is smaller than vuv_threshold.
    #
    # Outputs: 'f0': Estimated fundamental frequency (in Hz) for the given
    # frame.

    frame = np.array(frame)
    frame = frame - np.mean(frame)  # Remove mean to omit effect of DC component
    # Number of autocorrelation lag samples corresponding to f0_min (i.e. maximum
    # period length)
    max_lag = None  # This will be used as an index so you must make sure the variable is an integer
    # Number of autocorrelation lag samples corresponding to f0_max (i.e. minimum
    # period length)
    min_lag = None  # This will be used as an index so you must make sure the variable is an integer

    # Compute autocorrelation sequence up to max_lag
    r = np.correlate(frame, frame - lagged, mode="full")

    # Locate autocorrelation peak and its amplitude between min_lag and max_lag

    if ac_peak_val > vuv_threshold:
        f0 = None  # Compute f0 from obtained lag value
    else:
        f0 = 0

    return f0, ac_peak_val


def fundf_cepstrum(frame, fs, f0_min, f0_max, vuv_threshold):
    # EX2_FUNDF_CEPSTRUM Fundamental frequency estimation with the cepstrum method.
    # Inputs:   'frame': Windowed signal segment
    #           'fs': sampling frequency
    #           'f0_min' and 'f0_max': Given in Hz, represent the search space
    #           for the F0 values
    #           'vuv_threshold': Heuristic value that acts as a classifier
    #           between voiced and unvoiced frames. A frame is classified as unvoiced if the strength of the
    #           cepstral peak is smaller than vuv_threshold.
    #
    # Outputs:  'f0': Estimated fundamental frequency (in Hz) for the given
    # frame.
    #           'ceps_peak_val': The amplitude of the cepstral peak value
    frame = np.array(frame)

    # Number of autocorrelation lag samples corresponding to f0_min (i.e. maximum
    # period length)
    max_lag = None  # This will be used as an index so you must make sure the variable is an integer
    # Number of autocorrelation lag samples corresponding to f0_max (i.e. minimum
    # period length)
    min_lag = None  # This will be used as an index so you must make sure the variable is an integer

    eps = 0.00001  # Add this to the power spectrum to ensure values are above zero for log function

    # Compute real cepstrum of frame
    c = None

    # Locate cepstral peak and its amplitude between min_lag and max_lag

    if cepstral_peak_val > vuv_threshold:
        f0 = None  # Compute f0 from obtained quefrency value
    else:
        f0 = 0

    return f0, cepstral_peak_val
