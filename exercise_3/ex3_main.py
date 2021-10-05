# ELEC-E5500 Speech Processing -- Autumn 2021 Python 3.5 Exercise 3:
# Voice activity detection
import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import ex1_windowing_solution as win
import ex2_fundf_functions_solution as fundf
import ex3_functions as funcs
import matplotlib.pyplot as plt

##Section 0: Use the code from exercise 1 to read and window the analyzed sample:
# 0.1 Read the audio file and sampling rate
file_path = path.join(".", "Sounds")
sound_file = path.join(file_path, "SX83.wav")
Fs_target = 16000

Fs, in_sig = wav.read(sound_file)

# Transform signal from int16 (-32768 to 32767) to float32 (-1,1)
if type(in_sig[0]) == np.int16:
    in_sig = np.divide(in_sig, 32768, dtype=np.float32)

# 0.2 Make sure the sampling rate is 16kHz
if not (Fs == Fs_target):
    in_sig = sig.resample_poly(in_sig, Fs_target, Fs)
    Fs = Fs_target

# 0.3 Split the data sequence into windows. Use ex1_windowing
frame_length = int(np.around(0.025 * Fs))  # 25ms in samples
hop_size = int(np.around(0.0125 * Fs))  # 12.5 ms in samples (50% overlap)
window_types = ("rect", "hann", "cosine", "hamming")
frame_matrix = win.ex1_windowing_solution(
    in_sig, frame_length, hop_size, window_types[1]
)  # Windowing

##Section 1: acquire parameters for VUV detection:

# 1.1 Implement zero-crossing rate (ZCR) computation
zcr_vec = np.zeros((1, len(frame_matrix[0])))
for iFrame in range(len(frame_matrix[0])):
    zcr_vec[0, iFrame] = funcs.zcr(frame_matrix[:, iFrame])

# 1.2 Implement energy computation
energy_vec = np.zeros((1, len(frame_matrix[0])))
for iFrame in range(len(frame_matrix[0])):
    energy_vec[0, iFrame] = funcs.energy(frame_matrix[:, iFrame])

# 1.3 Implement one-lag autocorrelation computation
ac_vec = np.zeros((1, len(frame_matrix[0])))
for iFrame in range(len(frame_matrix[0])):
    ac_vec[0, iFrame] = funcs.one_lag_autocorrelation(frame_matrix[:, iFrame])

# 1.4 Use the functions from Exercise 2 to obtain peak values of the fundamental frequency:
f0_max = 180
f0_min = 80

ac_peak_vec = np.zeros((1, len(frame_matrix[0])))
ceps_peak_vec = np.zeros((1, len(frame_matrix[0])))

for iFrame in range(len(frame_matrix[0])):
    _, ac_peak_vec[0, iFrame] = fundf.fundf_autocorr_solution(
        frame_matrix[:, iFrame], Fs, f0_min, f0_max, 0
    )
    _, ceps_peak_vec[0, iFrame] = fundf.fundf_cepstrum_solution(
        frame_matrix[:, iFrame], Fs, f0_min, f0_max, 0
    )

##Section 2: Train a perceptron model with the computed input parameters for VAD.

# Load target output
vad_target = funcs.read_targets()

# 2.1 Concatenate input vectors
vad_input = np.concatenate((zcr_vec, energy_vec, ac_vec, ac_peak_vec, ceps_peak_vec))

# 2.2 Add deltas and delta-deltas to input
vad_input = funcs.add_deltas_deltadeltas(vad_input)

# 2.3 Normalize each input parameter to zero-mean and unit variance vectors
vad_input = funcs.normalize(vad_input)

# Add bias vector to input parameters
vad_input = vad_input + 0.5
# Train a perceptron model (linear classifier with output non-linearity)
w_perceptron = funcs.perceptron_training(vad_input, vad_target)
# Train a linear classifier
w_linear = np.matmul((vad_target - 0.5), np.linalg.pinv(vad_input))

# 2.4 Use the obtained models to make the predictions

thresh_perceptron = 0.0
vad_perceptron = (
    vad_input.T @ w_perceptron.T
)  # VAD Classification of the utterance based on the trained perceptron model with vad_input as the input matrix
activity_perceptron = np.where(vad_perceptron > thresh_perceptron, 1, 0)
error_perceptron = (
    np.sum(np.logical_xor(activity_perceptron, vad_target.T)) / vad_target.T.shape[0]
)  # Classification error
print(f"Error of perceptron: {error_perceptron * 100:.2f}%")

thresh_linear = 0.0
vad_linear = (
    vad_input.T @ w_linear.T
)  # VAD Classification of the utterance based on the linear model with vad_input as the input matrix
activity_linear = np.where(vad_linear > thresh_linear, 1, 0)
error_linear = (
    np.sum(np.logical_xor(activity_linear, vad_target.T)) / vad_target.T.shape[0]
)  # Classification error
print(f"Error of classifier: {error_linear * 100:.2f}%")

##Section 3: Visualize your results
plt.figure(1)

# 3.1 Plot the input vectors
plt.subplot(3, 1, 1)

# (zcr_vec, energy_vec, ac_vec, ac_peak_vec, ceps_peak_vec)
plt.plot(vad_input.T)
plt.autoscale(enable=True, axis="both", tight=True)
plt.title("Computed parameters (vad input)")

# 3.2 Plot the classifier activation values for each frame (without classification)

plt.subplot(3, 1, 2)

plt.plot(vad_target[0])
plt.plot(vad_perceptron, "--")
plt.plot(vad_linear, "-.")
plt.autoscale(enable=True, axis="both", tight=True)
plt.legend(("Target", "Perceptron", "Linear"))
plt.title("Classifier activations")

# 3.3 Plot the target vector,m and obtained classification results for both classifiers
plt.subplot(3, 1, 3)

plt.plot(vad_target[0])
plt.plot(activity_perceptron, "--")
plt.plot(activity_linear, "-.")
plt.axis([1, 214, -0.1, 1.1])
plt.legend(("Target", "Perceptron", "Linear"))
plt.title("Classifier outputs")
plt.show()
## Section 4: Experiment with different input parameters.
# a) What is the single best parameter for VAD classification in the test utterance?
#    Cepstral vector gives the best results when testing the features.

# b) What is the worst performing single parameter?
#    Autocorrelation vector gives the worst performance.

# c) What other features could be useful in VAD?
#    Magnitude spectrum and perhaps a spectrogram in tensor form.
