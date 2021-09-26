import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import ex1_windowing_solution as win

import ex2_fundf_functions as fundf
import matplotlib.pyplot as pl

"""
ELEC-E5500 Speech Processing -- Autumn 2021 Python Exercise 2:
Fundamental frequency estimation main.
"""

# Section 1: Use the code from Exercise 1 to read and window the analyzed sample:
# 1.1. Read the audio file and sampling rate

file_path = path.join(".", "Sounds")
sound_file = path.join(file_path, "bernard_speech.wav")
Fs_target = 16000

Fs, in_sig = wav.read(sound_file)  # Read audio file

# Scipy wav.read reads wav files as 16 bit integers from -32768 to 32767. We need to transform it into floats from -1 to 1. This will avoid further normalization problems.
if type(in_sig[0]) == np.int16:
    data = np.divide(in_sig, 32768, dtype=np.float32)

# 1.2. Make sure the sampling rate is 16kHz. Resample if not 16kHz

if Fs != Fs_target:
    in_sig = sig.resample(in_sig, int(np.round(Fs_target * (in_sig.shape[0] / Fs))))
    Fs = Fs_target

# 1.3. Split the data sequence into windows. Implement windowing.py

frame_length = np.int(np.around(0.025 * Fs))  # 25ms in samples
hop_size = np.int(np.around(0.0125 * Fs))  # 12.5 ms in samples (50% overlap)
window_types = ("rect", "hann", "cosine", "hamming")
frame_matrix = win.ex1_windowing_solution(
    in_sig, frame_length, hop_size, window_types[3]
)  # Windowing

## Section 2: Fundamental frequency estimation with the autocorrelation method
# 2.1. Define minimum and maximum values for the F0 search range, and the
# threshold value for Voiced/Unvoiced decision.
f0_max = 155
f0_min = 85
vuv_threshold_ac = None

# 2.2 Write a loop through frame_matrix that calls the function ex2_fundf_autocorr to obtain
# the F0 estimates for each frame

f0vec_ac = np.zeros(
    (len(frame_matrix[0]), 1)
)  # Allocate f0 vector for autocorrelation method
ac_peak_vec = np.zeros(
    (len(frame_matrix[0]), 1)
)  # Allocate vector for ac peak amplitude vector
for iFrame in range(len(frame_matrix[0])):
    (f0vec_ac[iFrame], ac_peak_vec[iFrame],) = fundf.fundf_autocorr(
        iFrame, Fs, f0_min, f0_max, vuv_threshold_ac
    )  # Estimate fundamental frequency using autocorrelation method
# Section 3: Fundamental frequency estimation with the cepstrum method

vuv_threshold_ceps = None
f0vec_ceps = np.zeros(
    (len(frame_matrix[0]), 1)
)  # Allocate f0 vector for cepstrum method
ceps_peak_vec = np.zeros(
    (len(frame_matrix[0]), 1)
)  # Allocate vector for cepstral peak vector
for iFrame in range(len(frame_matrix[0])):
    (f0vec_ceps[iFrame], ceps_peak_vec[iFrame],) = fundf.fundf_cepstrum(
        iFrame, Fs, f0_min, f0_max, vuv_threshold_ceps
    )  # Estimate fundamental frequency using cepstrum method

# Section 4: Test & visualize your results
Nfft = 1024
pl.figure(1)

# 4.1. Plot the spectrogram of the original signal as in Ex 1.4.4.
pl.subplot(3, 1, 1)


# 4.2. Plot the estimated F0 vectors. Report F0max and F0min within the title
pl.subplot(3, 1, 2)


# 4.3. Plot the peak amplitudes of the cepstral peak and the autocorrelation peak
pl.subplot(3, 1, 3)

pl.show()
# 4.4 Experiment with the parameters.
# a) How does tuning these parameters affect the autocorrelation method?
# Report your findings regarding at least:
#       i) Frame length
#       ii) Windowing function
#       iii) F0 search range
#       iv) Voicing threshold value
# b) How about the cepstrum method?
#       i) Frame length
#       ii) Windowing function
#       iii) F0 search range
#       iv) Voicing threshold value
