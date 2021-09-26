"""
EX1_WINDOWING Based on the input parameters, generate a n x m matrix of windowed
frames, with n corresponding to frame_length and m corresponding to number
of frames. The first frame starts at the beginning of the data.
"""

import os
import numpy as np


def cosine(N):
    return np.array([np.cos((np.pi * x) / N - np.pi / 2) for x in range(0, N)])


def ex1_windowing(data, frame_length, hop_size, windowing_function):
    data = np.array(data)
    number_of_frames = int(np.floor((data.shape[0] - frame_length) / hop_size)) + 1
    # Calculate the number of frames using the presented formula
    frame_matrix = np.zeros((frame_length, number_of_frames))

    if windowing_function == "rect":
        window = np.ones([frame_length,])  # Implement this
    elif windowing_function == "hann":
        window = np.hanning(frame_length)  # Implement this
    elif windowing_function == "cosine":
        window = cosine(frame_length)  # Implement this
    elif windowing_function == "hamming":
        window = np.hamming(frame_length)  # Implement this
    else:
        os.error("Windowing function not supported")

    ## Copy each frame segment from data to the corresponding column of frame_matrix.
    ## If the end sample of the frame segment is larger than data length,
    ## zero-pad the remainder to achieve constant frame length.
    ## Remember to apply the chosen windowing function to the frame!
    for i in range(number_of_frames):
        frame = np.zeros(frame_length)  # Initialize frame as zeroes

        # Implement the rest!
        start = i * hop_size

        end = min(start + frame_length, data.shape[0])

        signal_crop = data[start:end]

        frame[: end - start] = signal_crop
        frame *= window

        frame_matrix[:, i] = frame  # Copy frame to frame_matrix
    return frame_matrix
