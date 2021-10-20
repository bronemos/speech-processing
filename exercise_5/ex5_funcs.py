import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt

# Provided functions
def getWindow(frame_length, windowing_type):
    if windowing_type == "rect":
        window = np.ones(frame_length)
    elif windowing_type == "hann":
        window = np.hanning(frame_length)
    elif windowing_type == "cosine":
        window = np.sqrt(np.hanning(frame_length))
    elif windowing_type == "hamming":
        window = np.hamming(frame_length)
    else:
        print("Windowing function not supported")
        exit()

    return window.reshape(-1, 1)


def WGN(data_len, snr):
    # Add white gaussian noise to the signal with the defined snr.

    noise = np.random.normal(0, 1, data_len)
    pow_ratio = np.power(10, (snr / 20))
    noise_red = noise * pow_ratio

    return noise_red


def read_targets():
    # Target files must be in the same directory
    with open("output_targets", "r") as f:
        data = f.read()

    targets = np.array([int(i) for i in data.split()])
    targets = targets.reshape((1, len(targets)))
    return targets


# Complete these functions


def noiseEst(data_matrix, est_type):
    # If estimation type is ideal noise, return back the true noise value. Else, if estimation type is avg_noise_model, get the average noise model by computing the mean of the noise type. The dimesions of 'ret' should be same as the dimensions of the input data_matrix
    if est_type == "avg_noise_model":
        noise_mat = data_matrix
        noise_freq = np.fft.fft(noise_mat, axis=0)
        # Get the average-noise model
        avg_noise = np.reshape(np.mean(noise_freq, axis=0), [-1, 1])

        # Return model of the same dimensions as input noise matrix
        ret = matlib.repmat(avg_noise, 1, data_matrix.shape[0]).T

        assert ret.shape == data_matrix.shape
    elif est_type == "ideal_noise":
        # Get the ideal noise
        noise_mat = data_matrix
        noise_freq = np.fft.fft(noise_mat, axis=0)
        ret = noise_freq
        assert ret.shape == data_matrix.shape
    else:
        print("Wrong noise type")
        exit()

    return ret


def spectralSub(frame_matrix, hop_size, window_type, noise_est, original_signal_length):
    """
    Return the enhacned signal after spectral subtraction
    """

    # Setting initial variables
    frame_length = len(frame_matrix)
    fftlen = int(np.around(frame_length / 2) + 1)
    hwin = getWindow(frame_length, window_type)
    xest = np.zeros((original_signal_length, 1))

    # Performing enhancement for each frame

    for winix in range(len(frame_matrix[0])):
        start = int(winix * hop_size)
        stop = int(np.minimum(start + frame_length, original_signal_length))

        # Applying FFT and leaving out the symmetrical second half (or use numpy rfft)
        spectrum = np.fft.rfft(frame_matrix[:, winix])

        # Applying filtering - Spectral Subtraction
        spectral_subtraction = np.subtract(
            np.abs(spectrum) ** 2, np.abs(noise_est[:fftlen, winix]) ** 2,
        )
        spectral_subtraction = [x if x > 0 else 0 for x in spectral_subtraction]
        enhancement = np.sqrt(spectral_subtraction / np.abs(spectrum) ** 2)
        spectrum_enhanced = spectrum * enhancement

        # Reconstruction and inverse-FFT
        xwinest = np.real(
            np.reshape(np.fft.irfft(spectrum_enhanced), [frame_length, 1]) * hwin
        )
        # Overlap-add to get back the entire signal
        xest[start:stop] += xwinest

    return xest


def wiener(frame_matrix, hop_size, window_type, noise_est, original_signal_length):
    """
    Return the enhacned signal after Wiener filtering
    """

    # Setting initial variables
    frame_length = len(frame_matrix)
    fftlen = int(np.around(frame_length / 2) + 1)
    hwin = getWindow(frame_length, window_type)
    xest = np.zeros((original_signal_length, 1))

    # Performging enhancement for each frame

    for winix in range(len(frame_matrix[0])):
        start = int(winix * hop_size)
        stop = int(np.minimum(start + frame_length, original_signal_length))

        # Applying FFT and leaving out the symmetrical second half (or use numpy rfft)
        spectrum = np.fft.rfft(frame_matrix[:, winix])

        # Applying filtering - Wiener
        wiener = np.subtract(
            np.abs(spectrum) ** 2, np.abs(noise_est[:fftlen, winix]) ** 2
        )
        wiener = [x if x > 0 else 0 for x in wiener]
        enhancement = wiener / np.abs(spectrum) ** 2
        spectrum_enhanced = spectrum * enhancement

        # Reconstruction and inverse-FFT
        xwinest = np.reshape(np.fft.irfft(spectrum_enhanced), [frame_length, 1]) * hwin
        # Overlap-add to get back the entire signal
        xest[start:stop] += xwinest
    return xest


def linear(frame_matrix, hop_size, window_type, noise_est, original_signal_length):
    """
    Return the enhacned signal after linear filtering
    """

    # Setting initial variables
    frame_length = len(frame_matrix)
    fftlen = int(np.around(frame_length / 2) + 1)
    hwin = getWindow(frame_length, window_type)
    xest = np.zeros((original_signal_length, 1))

    # Performging enhancement for each frame

    for winix in range(len(frame_matrix[0])):
        start = int(winix * hop_size)
        stop = int(np.minimum(start + frame_length, original_signal_length))

        # Applying FFT and leaving out the symmetrical second half (or use numpy rfft)
        spectrum = np.fft.rfft(frame_matrix[:, winix])

        # Applyin filtering - Linear
        linear = np.subtract(np.abs(spectrum), np.abs(noise_est[:fftlen, winix]))
        condition_check = np.subtract(
            np.abs(spectrum) ** 2, np.abs(noise_est[:fftlen, winix]) ** 2
        )  # **2 and subtracted needs to be > 0 - used solely for checking the condition
        linear = [x if check > 0 else 0 for x, check in zip(linear, condition_check)]
        enhancement = linear / np.abs(spectrum)
        linear_enhanced = spectrum * enhancement

        # Reconstruction and inverse-FFT
        xwinest = np.reshape(np.fft.irfft(linear_enhanced), [frame_length, 1]) * hwin

        # Overlap-add to get back the entire signal
        xest[start:stop] += xwinest

    return xest


def vadEnhance(
    frame_matrix, hop_size, window_type, noise_est, original_signal_length, vad_target
):
    """
    Return the enhacned signal after Wiener filtering with VAD trigger
    """

    # Setting initial variables
    frame_length = len(frame_matrix)
    fftlen = int(np.around(frame_length / 2) + 1)
    hwin = getWindow(frame_length, window_type)
    xest = np.zeros((original_signal_length, 1))

    # Performging enhancement for each frame

    for winix in range(len(frame_matrix[0])):
        start = int(winix * hop_size)
        stop = int(np.minimum(start + frame_length, original_signal_length))

        # Applying FFT and leaving out the symmetrical second half (or use numpy rfft)
        spectrum = np.fft.rfft(frame_matrix[:, winix])

        # Applying filtering - VAD enhancement
        if vad_target[winix]:
            vad = np.subtract(
                np.abs(spectrum) ** 2, np.abs(noise_est[:fftlen, winix]) ** 2
            )
            enhancement = vad / np.abs(spectrum) ** 2
            vad_enhanced = spectrum * enhancement

            # Reconstruction and inverse-FFT
            xwinest = np.reshape(np.fft.irfft(vad_enhanced), [frame_length, 1]) * hwin

        else:
            xwinest = np.zeros((frame_length, 1))

        # Overlap-add to get back the entire signal
        xest[start:stop] += xwinest

    return xest


def snrGlb(clean, enhanced):
    if not (len(clean) == len(enhanced)):
        print("Length of signals do not match")
        exit()

    # Compute the global SNR in dB
    clean = np.reshape(clean, -1)
    enhanced = np.reshape(enhanced, -1)
    ret = 10 * np.log10(
        np.sum(np.abs(clean) ** 2) / np.sum(np.abs(clean - enhanced) ** 2)
    )

    return ret


def snrSeg(clean, enhanced):
    if not (len(clean) == len(enhanced)):
        print("Length of signals do not match")
        exit()

    # Compute the segmental-SNR in dB
    snr_seg = np.zeros(len(clean[0]))
    for winix in range(len(clean[0])):
        spectra_clean = np.fft.rfft(clean[:, winix])
        spectra_enhanced = np.fft.rfft(enhanced[:, winix])
        snr = 10 * np.log10(
            np.sum(np.abs(spectra_clean) ** 2)
            / np.sum(np.abs(spectra_clean - spectra_enhanced) ** 2)
        )
        snr_seg[winix] = snr

    ret = snr_seg

    return ret
