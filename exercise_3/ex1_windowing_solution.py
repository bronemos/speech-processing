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
EX1_WINDOWING Based on the input parameters, generate a n x m matrix of windowed
frames, with n corresponding to frame_length and m corresponding to number
of frames. The first frame starts at the beginning of the data.
"""
import os
import numpy as np


def ex1_windowing_solution(data, frame_length, hop_size, windowing_function):
    data = np.array(data)
    number_of_frames = 1 + int(
        np.floor((len(data) - frame_length) / hop_size)
    )  # Calculate the number of frames using the presented formula
    frame_matrix = np.zeros((frame_length, number_of_frames))
    if windowing_function == "rect":
        window = np.ones((frame_length,))  # Implement this
    elif windowing_function == "hann":
        window = np.hanning(frame_length)  # Implement this
    elif windowing_function == "cosine":
        window = np.sqrt(np.hanning(frame_length))  # Implement this
    elif windowing_function == "hamming":
        window = np.hamming(frame_length)  # Implement this
    else:
        os.error("Windowing function not supported")

    ## Copy each frame segment from data to the corresponding column of frame_matrix.
    ## If the end sample of the frame segment is larger than data length,
    ## zero-pad the remainder to achieve constant frame length.
    ## Remember to apply the chosen windowing function to the frame!
    for i in range(number_of_frames):
        start = i * hop_size
        stop = np.minimum(start + frame_length, len(data))
        frame = np.zeros(frame_length)

        frame[0 : stop - start] = data[start:stop]
        frame_matrix[:, i] = np.multiply(window, frame)
    return frame_matrix

