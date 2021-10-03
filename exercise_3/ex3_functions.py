# Functions for exercise 3 of the course Speech Processing course
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
import scipy.signal as sig
import numpy as np

#########################################################################################################
# Already made functions to use


def read_targets():
    # Target files must be in the same directory
    with open("output_targets", "r") as f:
        data = f.read()

    targets = np.array([int(i) for i in data.split()])
    targets = targets.reshape((1, len(targets)))
    return targets


def normalize(M):
    # Normalize (zero-mean and unit variance) the values of input matrix M for each row.

    output = np.zeros(M.shape)

    for i in range(len(M)):
        tmp = M[i, :]
        tmp = tmp - np.mean(tmp)
        tmp = tmp / np.linalg.norm(tmp, ord=2)
        output[i, :] = tmp

    return output


def perceptron_training(vad_input, outputs):

    MAX_ITER = 100000
    w = np.zeros((1, len(vad_input)))
    y = np.zeros((1, len(outputs)))

    # Training
    for iter in range(MAX_ITER):
        # Compute weight gradient
        dw = np.matmul((outputs - y), vad_input.T) / len(vad_input[0])

        # update weights
        w = w + dw

        # Apply non-linearity
        y = np.matmul(w, vad_input) > 0

    print("Perceptron training complete after " + str(iter) + " iterations.")

    return w


##########################################################################################################

# Functions to complete in the exercise


def zcr(frame):
    # Count the number of times that a signal crosses the zero-line
    # Inputs: frame: the input signal frame
    # Outputs: zcr: The zero-crossing rate of a zero-mean frame.
    # (I.e. Remember to remove the mean from the frame!!)

    frame = np.array(frame)
    frame = frame - np.mean(frame)
    zcr = np.count_nonzero(np.sign(frame[:-1] @ frame[1:].T) == -1)  # Complete

    return zcr


def one_lag_autocorrelation(frame):
    # Returns the 1-lag autocorrelation coefficient of input sequence.
    # Removes mean and normalizes the coefficients based on the zero-lag coefficient.
    # Inputs: frame: the input signal frame
    # Outputs: val: The one-lag autocorrelation coefficient

    frame = np.array(frame)
    frame = frame - np.mean(frame)
    val = np.dot(frame[:-1], frame[1:])  # Complete

    return val


def energy(frame):
    # Compute the energy of a given frame (with mean removed)
    # Inputs: frame: the input signal frame
    # Outputs: energy: Frame energy

    energy = (
        np.sqrt(np.sum(np.power(frame - np.mean(frame), 2))) / frame.shape[0]
    )  # Complete

    return energy


def add_deltas_deltadeltas(vad_input):
    # Add delta and delta-delta features to features of input matrix. Rows represent features, and columns represent frames.
    # Inputs: vad_input: m x n matrix with m features and n frames
    # Outputs: output: 3*m x n matrix whose rows contain original features, delta and delta-delta features

    # Delta and Delta-delta filters:
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    filt_dx = sig.lfilter([-1, 0, 1], [])
    filt_ddx = sig.lfilter([1, -2, 1], [])

    output = None

    return output
