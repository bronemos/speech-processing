import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import ex1_windowing_solution as win
import ex5_funcs as funcs
import matplotlib.pyplot as plt


# Section 0: Use the code from exercise 1 to read and window the analyzed sample:

# Split the noisy data_clean sequence into windows. You can use ex1_windowing_solution

# 0.1 Read the audio file and sampling rate
file_path = path.join(".", "Sounds")
audio_file = path.join(file_path, "SX83.wav")
Fs_target = 16000
Fs, data_clean = wav.read(audio_file)

# Transform signal from int16 (-32768 to 32767) to float32 (-1,1)
if type(data_clean[0]) == np.int16:
    data_clean = np.divide(data_clean, 32768, dtype=np.float32)

# 0.2 Make sure the sampling rate is 16kHz
if not (Fs == Fs_target):
    data_clean = sig.resample_poly(data_clean, Fs_target, Fs)
    Fs = Fs_target

# Generate a noisy signal
# Generate white Gaussian noise of strength -35dB
data_noise = funcs.WGN(data_clean.shape[0], -35)
# Generate the noisy signal, where the noise is additive white Guassian
data_noisy = data_clean + data_noise

# 0.3 Split the data_clean sequence into windows. Use ex1_windowing
frame_length = int(np.around(0.025 * Fs))  # 25ms in samples
hop_size = int(np.around(0.0125 * Fs))  # 12.5 ms in samples (50% overlap)
window_types = ("rect", "hann", "cosine", "hamming")
win_i = 3
frame_matrix = win.ex1_windowing_solution(
    data_noisy, frame_length, hop_size, window_types[win_i]
)  # Windowing

# Section 1: Estimate noise model
estimation_types = ("ideal_noise", "avg_noise_model")
est_type = 0
# Window the generated white Gaussian noise signal/s for modelling:
frame_matrix_noise = win.ex1_windowing_solution(
    data_noise, frame_length, hop_size, window_types[win_i]
)
# Obtain the noise model which will be used later for noise reduction
noise_est = funcs.noiseEst(frame_matrix_noise, estimation_types[est_type])

# Section 2: Enhancement
# Perform spectral subtraction
enhanced_sig_specSub = funcs.spectralSub(
    frame_matrix, hop_size, window_types[win_i], noise_est, len(data_noisy)
)

# Perform Wiener filtering
enhanced_sig_wiener = funcs.wiener(
    frame_matrix, hop_size, window_types[win_i], noise_est, len(data_noisy)
)

# Perform linear filtering
enhanced_sig_linear = funcs.linear(
    frame_matrix, hop_size, window_types[win_i], noise_est, len(data_noisy)
)

# VAD based noise-reduction
output_targets = funcs.read_targets().reshape(-1, 1)

enhanced_sig_vad = funcs.vadEnhance(
    frame_matrix,
    hop_size,
    window_types[win_i],
    noise_est,
    len(data_noisy),
    output_targets,
)

## Section 3: Evaluation

# Compute the global SNR in dB for all the enhanced signals using spectral subtraction, Wiener filtering, linear filtering
SNR_global_noisy = funcs.snrGlb(data_clean, data_noisy)
SNR_global_ss = funcs.snrGlb(data_clean, enhanced_sig_specSub)
SNR_global_wie = funcs.snrGlb(data_clean, enhanced_sig_wiener)
SNR_global_linear = funcs.snrGlb(data_clean, enhanced_sig_linear)
SNR_global_vad = funcs.snrGlb(data_clean, enhanced_sig_vad)

print(f"{'Global SNR Noisy:':35s}{SNR_global_noisy:4f} dB")
print(f"{'Global SNR Spectral Subtraction:':35s}{SNR_global_ss:4f} dB")
print(f"{'Global SNR Wiener:':35s}{SNR_global_wie:4f} dB")
print(f"{'Global SNR Linear:':35s}{SNR_global_linear:4f} dB")
print(f"{'Global SNR VAD':35s}{SNR_global_vad:4f} dB")

# Segmental-SNR
# First, Window the clean and enhanced signals
frame_matrix_clean = win.ex1_windowing_solution(
    data_clean, frame_length, hop_size, window_types[win_i]
)
frame_matrix_enhSS = win.ex1_windowing_solution(
    enhanced_sig_specSub.reshape((-1,)), frame_length, hop_size, window_types[win_i]
)
frame_matrix_enhWie = win.ex1_windowing_solution(
    enhanced_sig_wiener.reshape((-1,)), frame_length, hop_size, window_types[win_i]
)
frame_matrix_enhLin = win.ex1_windowing_solution(
    enhanced_sig_linear.reshape((-1,)), frame_length, hop_size, window_types[win_i]
)
frame_matrix_enhvad = win.ex1_windowing_solution(
    enhanced_sig_vad.reshape((-1,)), frame_length, hop_size, window_types[win_i]
)

# Then compute the segmental SNR
SNR_seg_noisy = funcs.snrSeg(frame_matrix_clean, frame_matrix)
SNR_seg_ss = funcs.snrSeg(frame_matrix_clean, frame_matrix_enhSS)
SNR_seg_wie = funcs.snrSeg(frame_matrix_clean, frame_matrix_enhWie)
SNR_seg_linear = funcs.snrSeg(frame_matrix_clean, frame_matrix_enhLin)
SNR_seg_vad = funcs.snrSeg(frame_matrix_clean, frame_matrix_enhvad)

## Section 4: Plotting and visualization
plt.figure()

# Plot the noisy signal and the segmental SNRs from the 4 methods. Let the x-axis denote frames and y-axis denote the SNR in dB

plt.plot(SNR_seg_noisy)
plt.plot(SNR_seg_ss)
plt.plot(SNR_seg_wie)
plt.plot(SNR_seg_linear)
plt.plot(SNR_seg_vad)
plt.legend(("noisy", "spectral subtraction", "Wiener", "Linear", "vad"))
plt.ylabel("SNR [dB]")
plt.xlabel("Frame number")
plt.title("SNR comparison")
plt.show(block=False)
# Plot the spectrograms of the clean, noisy and the enhanced signals
# In this figure, plot the clean and noisy spectrograms
plt.figure()
eps = 0.000001  # Add before log10 to avoid zeros
Nfft = 1024

frame_matrix_clean_fft = 20 * np.log10(
    np.absolute(np.flipud(np.fft.rfft(frame_matrix_clean, axis=0, n=Nfft)) + eps)
)

frame_matrix_noisy_fft = 20 * np.log10(
    np.absolute(np.flipud(np.fft.rfft(frame_matrix, axis=0, n=Nfft)) + eps)
)

f_axis = np.array(range(int((Nfft / 2) + 1))) / (Nfft / 2) * Fs / 2
plt.subplot(2, 1, 1)
plt.imshow(frame_matrix_clean_fft, aspect="auto")
plt.title("Spectrogram-Clean")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Frame number")

plt.subplot(2, 1, 2)
plt.imshow(frame_matrix_noisy_fft, aspect="auto")
plt.title("Spectrogram-Noisy")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Frame number")
plt.show(block=False)
# In this figure, plot the spectrograms of all the 4 enhanced signals
plt.figure()
# Plot the enhanced signal with spectral subtraction
plt.subplot(2, 2, 1)

frame_matrix_enhSS_fft = 20 * np.log10(
    np.absolute(np.flipud(np.fft.rfft(frame_matrix_enhSS, axis=0, n=Nfft)) + eps)
)
plt.imshow(frame_matrix_enhSS_fft, aspect="auto")

plt.title("Spectrogram-Spectral Subtraction")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Frame number")
# Plot the enhanced signal with Wiener filter
plt.subplot(2, 2, 2)

frame_matrix_enhWie_fft = 20 * np.log10(
    np.absolute(np.flipud(np.fft.rfft(frame_matrix_enhWie, axis=0, n=Nfft)) + eps)
)
plt.imshow(frame_matrix_enhWie_fft, aspect="auto")

plt.title("Spectrogram-Wiener filter")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Frame number")
# Plot the enhanced signal with linear method
plt.subplot(2, 2, 3)

frame_matrix_enhLin_fft = 20 * np.log10(
    np.absolute(np.flipud(np.fft.rfft(frame_matrix_enhLin, axis=0, n=Nfft)) + eps)
)
plt.imshow(frame_matrix_enhLin_fft, aspect="auto")

plt.title("Spectrogram-Linear")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Frame number")
# Plot the enhanced signal with VAD method
plt.subplot(2, 2, 4)

frame_matrix_enhvad_fft = 20 * np.log10(
    np.absolute(np.flipud(np.fft.rfft(frame_matrix_enhvad, axis=0, n=Nfft)) + eps)
)
plt.imshow(frame_matrix_enhvad_fft, aspect="auto")

plt.title("Spectrogram-VAD")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Frame number")

plt.show()
