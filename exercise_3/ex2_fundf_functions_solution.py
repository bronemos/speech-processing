"""
Copyright 2021 Bernard Spiegl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
ELEC-E5500 Speech Processing -- Autumn 2021 Exercise 2 SOLUTION:
"""
import numpy as np


def fundf_autocorr_solution(frame, fs, f0_min, f0_max, vuv_threshold):
    # EX2_FUNDF_AUTOCORR Fundamental frequency estimation with the
    # autocorrelation method.
    # Inputs:   'frame': Windowed signal segment
    #           'fs': sampling frequency
    #           'f0_min' and 'f0_max': Given in Hz, represent the search space
    #           for the F0 values
    #           'vuv_threshold': Heuristic value that acts as a classifier
    #           between voiced and unvoiced frames. A frame is classified as unvoiced if the strength of the
    #           autocorrelation peak is smaller than vuv_threshold.
    #
    # Outputs: 'f0': Estimated fundametal frequency (in Hz) for the given
    # frame.
    frame = np.array(frame)
    frame = frame - np.mean(frame)  # Remove mean to omit effect of DC component
    # Number of autocorrelation lag samples corresponding to f0_min (i.e. maximum # period length)
    max_lag = np.int(np.ceil(fs / f0_min))
    # Number of autocorrelation lag samples corresponding to f0_max (i.e. minimum # period length)
    min_lag = np.int(np.ceil(fs / f0_max))
    # Compute autorrelation sequence up to max_lag
    r = np.correlate(frame, frame, mode="full")
    r = r / np.amax(np.absolute(r))
    r = r[(len(frame) - 1) :]
    # Locate autocorrelation peak and its amplitude between min_lag and max_lag
    r = r[min_lag : (max_lag + 1)]
    ac_peak_val = np.amax(r)
    ind = np.argmax(r)
    # ind = np.where(r == ac_peak_val)
    # ind = np.int(ind[0])
    if ac_peak_val > vuv_threshold:
        f0 = fs / (min_lag + ind)
    else:
        f0 = 0
    return f0, ac_peak_val


def fundf_cepstrum_solution(frame, fs, f0_min, f0_max, vuv_threshold):
    # EX2_FUNDF_CEPSTRUM Fundamental frequency estimation with the cepstrum method.
    # Inputs:   'frame': Windowed signal segment
    #           'fs': sampling frequency
    #           'f0_min' and 'f0_max': Given in Hz, represent the search space
    #           for the F0 values
    #           'vuv_threshold': Heuristic value that acts as a classifier
    #           between voiced and unvoiced frames. A frame is classified as unvoiced if the strength of the
    #           cepstral peak is smaller than vuv_threshold.
    #
    # Outputs:  'f0': Estimated fundametal frequency (in Hz) for the given
    # frame.
    #           'ceps_peak_val': The amplitude of the cepstral peak value
    frame = np.array(frame)

    # Number of autocorrelation lag samples corresponding to f0_min (i.e. maximum
    # period length)
    max_lag = np.int(np.ceil(fs / f0_min))
    # Number of autocorrelation lag samples corresponding to f0_max (i.e. minimum
    # period length)
    min_lag = np.int(np.floor(fs / f0_max))

    eps = 0.00001  # Add this to the power spectrum to ensure values are above zero for log function

    # Compute real cepstrum of frame
    c = np.real(
        np.fft.ifft(np.log10(np.absolute(np.power(np.fft.fft(frame), 2) + eps)))
    )

    # Locate cepstral peak and its amplitude between min_lag and max_lag
    c = c[min_lag:max_lag]

    cepstral_peak_val = np.amax(np.absolute(c))
    ind = np.argmax(c)
    # ind = np.where(c == cepstral_peak_val)
    # ind = np.int(ind[0])

    if cepstral_peak_val > vuv_threshold:
        f0 = fs / (min_lag + ind)
    else:
        f0 = 0

    return f0, cepstral_peak_val
