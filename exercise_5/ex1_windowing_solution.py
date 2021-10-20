"""
EX1_WINDOWING Based on the input parameters, generate a n x m matrix of windowed
frames, with n corresponding to frame_length and m corresponding to number
of frames. The first frame starts at the beginning of the data.
"""
import os
import numpy as np


def ex1_windowing_solution(data, frame_length, hop_size, windowing_function):
    data = np.array(data)
    number_of_frames = 1 + np.int(np.floor((len(data) - frame_length) / hop_size))
    frame_matrix = np.zeros((frame_length, number_of_frames))

    if windowing_function == "rect":
        window = np.ones(frame_length)
    elif windowing_function == "hann":
        window = np.hanning(frame_length)
    elif windowing_function == "cosine":
        window = np.sqrt(np.hanning(frame_length))
    elif windowing_function == "hamming":
        window = np.hamming(frame_length)
    else:
        os.error("Windowing function not supported")

    for i in range(number_of_frames):
        start = i * hop_size
        stop = np.minimum(start + frame_length, len(data))

        frame = np.zeros(frame_length)

        frame[0 : stop - start] = data[start:stop]
        frame_matrix[:, i] = np.multiply(window, frame)
    return frame_matrix
