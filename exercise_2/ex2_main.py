import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import ex1_windowing_solution as win

import ex2_fundf_functions_solution as fundf
import matplotlib.pyplot as pl

"""
ELEC-E5500 Speech Processing -- Autumn 2021 Python Exercise 2:
Fundamental frequency estimation main.
"""

# Section 1: Use the code from Exercise 1 to read and window the analyzed sample:
# 1.1. Read the audio file and sampling rate

file_path = path.join(".", "Sounds")
sound_file = path.join(file_path, "bernard_speech2.wav")
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

frame_length = int(np.around(0.025 * Fs))  # 25ms in samples
hop_size = int(np.around(0.0125 * Fs))  # 12.5 ms in samples (50% overlap)
window_types = ("rect", "hann", "cosine", "hamming")
frame_matrix = win.ex1_windowing(
    in_sig, frame_length, hop_size, window_types[3]
)  # Windowing


## Section 2: Fundamental frequency estimation with the autocorrelation method
# 2.1. Define minimum and maximum values for the F0 search range, and the
# threshold value for Voiced/Unvoiced decision.
f0_max = 180
f0_min = 80
vuv_threshold_ac = 0.5

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
        frame_matrix[:, iFrame], Fs, f0_min, f0_max, vuv_threshold_ac
    )  # Estimate fundamental frequency using autocorrelation method
# Section 3: Fundamental frequency estimation with the cepstrum method

vuv_threshold_ceps = 0.073
f0vec_ceps = np.zeros(
    (len(frame_matrix[0]), 1)
)  # Allocate f0 vector for cepstrum method
ceps_peak_vec = np.zeros(
    (len(frame_matrix[0]), 1)
)  # Allocate vector for cepstral peak vector
for iFrame in range(len(frame_matrix[0])):
    (f0vec_ceps[iFrame], ceps_peak_vec[iFrame],) = fundf.fundf_cepstrum(
        frame_matrix[:, iFrame], Fs, f0_min, f0_max, vuv_threshold_ceps
    )  # Estimate fundamental frequency using cepstrum method

# Section 4: Test & visualize your results
Nfft = 1024
pl.figure(1)

# 4.1. Plot the spectrogram of the original signal as in Ex 1.4.4.
pl.subplot(3, 1, 1)
f_axis = np.divide(range(int(Nfft / 2) + 1), (Nfft / 2) / (Fs / 2))
frame_matrix_fft = np.fft.rfft(frame_matrix, axis=0, n=Nfft)
frame_matrix_fft = 20 * np.log10(np.absolute(np.flipud(frame_matrix_fft)))
pl.imshow(frame_matrix_fft, aspect="auto")
ytickpos = np.flipud(
    [0, int(Nfft / 8), int(Nfft / 4), int(Nfft * 3 / 8), int(Nfft / 2)]
)
pl.yticks([0, Nfft / 8, Nfft / 4, Nfft * 3 / 8, Nfft / 2], f_axis[ytickpos])
pl.title("Spectrogram")
pl.ylabel("Frequency (Hz)")
pl.xlabel("Frame number")


# 4.2. Plot the estimated F0 vectors. Report F0max and F0min within the title
pl.subplot(3, 1, 2)
pl.plot(f0vec_ac, label="AC")
pl.plot(f0vec_ceps, color="red", linestyle="dashed", label="Ceps")
pl.title(f"Estimated F0 contours: F0min={f0_min} F0max={f0_max}")
pl.ylabel("Frequency (Hz)")
pl.xlabel("Frame number")
pl.legend()


# 4.3. Plot the peak amplitudes of the cepstral peak and the autocorrelation peak
pl.subplot(3, 1, 3)
pl.plot(ac_peak_vec, label="AC")
pl.plot(ceps_peak_vec, color="red", linestyle="dashed", label="Ceps")
pl.title("Maximum peak strengths within the specified F0 search range")
pl.ylabel("Amplitude")
pl.xlabel("Frame number")
pl.legend()

pl.show()
# 4.4 Experiment with the parameters.
# a) How does tuning these parameters affect the autocorrelation method?
# Report your findings regarding at least:
#       i) Frame length
#       Frame length affect the interpretability of plots, if the frame length is too small the values on both plots oscillate quickly.
#       On the other hand, if the frame length is too big part of the information gets lost since the algorithm is taking bigger chunks of signal at a time.

#       ii) Windowing function
#       Using rectangular window causes a lot of spikes in fundamental frequency, other windows behave similarly to each other.

#       iii) F0 search range
#       If the search range is too small there are more spikes visible, with a range that has f0_max too high the estimates become highly innacurate and estimation oscillates significantly.

#       iv) Voicing threshold value
#       If the voicing threshold is too high no f0s are found and the plot sticks to 0, if the voicing threshold is too low the signal oscillates significantly.
#       This causes the estimation to show an f0 even when there is silence for example.

# b) How about the cepstrum method?
#       i) Frame length
#       Same as in autocorrelation method, it affects the interpretability of the plots. With a small frame length the cepstral method tends to return a constant f0
#       with amplitude severly oscillating. With an increase in the window size the sparsity of frequency peaks also increases and amplitude plot becomes harder to interpret.

#       ii) Windowing function
#       Rectangle underperforms same as in the AC method, cosine gives the smoothest result with other two windows following.

#       iii) F0 search range
#       Behaves almost exactly the same as the AC method, when the f0_max goes too high the peaks oscillate significantly.

#       iv) Voicing threshold value
#       In my case voicing threshold for this method is an order of magnitude smaller than the one in the AC method. Although the voicing threshold is smaller,
#       the effect it has on the estimation stays the same as in the AC method.

# Additional notes - Fundamental frequency estimations given by both methods perform better when there's a constant, stable voiced part of signal.
#                   E.g. at the end of my speech recording I let out "AAAA" sound at different frequencies which is nicely visible on the plot.
#                   Another thing I've noticed just from looking at the plots, when using cepstral method it seems to be "lagging" behind the AC method
#                   with its f0 estimations, i.e. the f0 reaches the same value but at a later time.
