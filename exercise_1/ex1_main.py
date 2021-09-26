#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELEC-E5500 Speech Processing -- Autumn 2020 Python Exercise 1:
Basics of speech processing and analysis in Python.

Recommended to use a virtual environment to have a clear management of the libraries used in the exercises.

Python version: 3.5 or higher

To make sure all the packages are up-to-date for the exercise, run the script Update_Packages_ex1.py.
"""
import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import ex1_windowing as win
import matplotlib.pyplot as pl

# custom function for creating a spectrogram


def fm_to_spectrogram(frame_matrix, hop_size):
    # frame_points = np.arange(0, frame_matrix.shape[0], step=hop_size)
    spectrums = []
    for frame in frame_matrix.T:
        spectrum = np.abs(np.fft.rfft(frame))
        spectrums.append(spectrum)
    spectrums = np.array(spectrums).T
    spectrums = 20 * np.log10(spectrums)
    return spectrums


# 1.1. Read the audio file bernard_speech.wav and sampling rate
file_path = path.join(".", "Sounds")
sound_file = path.join(file_path, "bernard_speech.wav")
Fs, in_sig = wav.read(sound_file)  # Read audio file


# 1.2. Make sure the sampling rate is 16kHz, resample if necessary
Fs_target = 16000
if Fs != Fs_target:
    in_sig = sig.resample(in_sig, int(np.round(Fs_target * (in_sig.shape[0] / Fs))))
    Fs = Fs_target


## 1.3. Split the data sequence into windows.
# Implement windowing function in ex1_windowing.py
frame_length = int(np.around(0.025 * Fs))  # 25ms in samples
hop_size = int(np.around(0.0125 * Fs))  # 12.5 ms in samples (50% overlap)
window_types = ("rect", "hann", "cosine", "hamming")
frame_matrix = win.ex1_windowing(
    in_sig, frame_length, hop_size, window_types[3]
)  # Windowing


# 1.4. Visualization. Create a new figure with three subplots.
pl.figure(1)
## 1.4.1. Plot the whole signal into subplot 1. Denote x-axis as time in seconds and y-axis as Amplitude.
### Set appropriate strings to title, xlabel and ylabel
pl.subplot(3, 1, 1)
pl.plot(in_sig)
pl.xticks(
    np.arange(0, in_sig.shape[0] + 1, step=Fs),
    np.arange(0, in_sig.shape[0] / Fs + 1, step=1.0),
)
pl.yticks(
    np.arange(-10000, 10000 + 1, step=2500), np.arange(-10000, 10000 + 1, step=2500)
)
pl.title("Original signal")
pl.xlabel("Time in seconds")
pl.ylabel("Amplitude")

## 1.4.2. Plot a VOICED frame from frame_matrix into subplot 2. Denote x-axis as milliseconds.
### Set appropriate strings to title, xlabel and ylabel
pl.subplot(3, 1, 2)

## randomly pick a voiced frame (frame with average above the given threshold)
voiced = False
while not voiced:
    frame_idx = np.random.randint(0, frame_matrix.shape[0])
    if np.abs(np.mean(frame_matrix[frame_idx])) > 80:
        voiced = True

## test frame
frame_idx = 35

pl.plot(frame_matrix.T[frame_idx])
pl.xticks(np.arange(0, 401, 16 * 5), np.arange(0, 26, 5))
pl.yticks()
pl.title("25 ms segment of a voiced frame")
pl.xlabel("Time in miliseconds")
pl.ylabel("Amplitude")

## 1.4.3. Plot the magnitude spectrum of the same frame as in 1.4.2. into
## subplot 3. Denote x-axis as Hz, and y-axis as decibels.
### Set appropriate strings to title, xlabel and ylabel
pl.subplot(3, 1, 3)
magnitude_spectrum = 20 * np.log10(np.abs(np.fft.rfft(frame_matrix.T[frame_idx])))
pl.plot(magnitude_spectrum)

## builtin matplotlib function (same result)
# pl.magnitude_spectrum(frame_matrix[frame_idx], Fs=Fs, scale="dB")

pl.xticks(np.arange(0, 201, step=25), np.arange(0, 9, step=1))
pl.title("Magnitude spectrum of the same frame")
pl.xlabel("Frequency (kHz)")
pl.ylabel("Amplitude (dB)")


pl.show(block=False)
## 1.4.4. Compute and plot the spectrogram of the whole signal into a new
## figure. Denote x-axis as frame number and y-axis as Frequency in Hz
### Set appropriate strings to title, xlabel and ylabel
pl.figure(2)
spectrogram = fm_to_spectrogram(frame_matrix, hop_size)
# pl.specgram(in_sig, Fs=Fs, scale="dB")
pl.imshow(spectrogram, origin="lower")
pl.yticks(np.arange(0, 201, 50), np.arange(0, 8001, 2000))


pl.xlabel("Frame number")
pl.ylabel("Frequency (Hz)")

pl.show()
