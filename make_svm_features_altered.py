"""Creates a folder "SVM features" containing feature arrays for use in classification by the SVM.

These are saved in .npy file format, and can be loaded using np.load("filename.npy") 

The dimensions of the saved arrays are (n_prompts * n_epochs * n_windowsXn_features)

The layout of the features in dim 2 is as follows (where d means delta, and dd double-delta):

[mean, absmean, maximum... d_mean, d_absmean, d_maximum... dd_mean, dd_absmean, dd_maximum ... dd_dfa]

To get a list of feature names (in order), use the function get_feats_list()

For convenience, we also save an array of the labels [goose, thought ... ], with dimension (n_prompts) 
in the same folder as the features. For most experiments, the labels will be identical. 

To use entropy/time series features, follow the installation instructions for this package on GitHub:
https://github.com/raphaelvallat/entropy


NOTE: this version of this code (JelleHubertse/Dissertation GitHub repo) was slightly edited for this dissertation. 
Changes made are the exlcusion of the enhancement feature and the added bandpassing function.
For the orignal code, check out the original FEIS repo here: https://github.com/scottwellington/FEIS

This code assumes the existence of a ./experiments directory. Directories inside the experiments directory represent experiments.
Experiments should be called 'Ex_<name>' (where <name> is the name of your experiment), should contain the respective (unzipped) csv files from which features should be extracted.

"""

import os
import os.path as op
import numpy as np
from scipy import integrate, stats
import re
import entropy
import sys
import mne

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


fs = 256  # sampling frequency, used in the calculation of the spectral entropy

# Features used in classification:


def mean(x):
    return np.mean(x)


def absmean(x):
    return np.mean(np.abs(x))


def maximum(x):
    return np.max(x)


def absmax(x):
    return np.max(np.abs(x))


def minimum(x):
    return np.min(x)


def absmin(x):
    return np.min(np.abs(x))


def minplusmax(x):
    return np.max(x) + np.min(x)


def maxminusmin(x):
    return np.max(x) - np.min(x)


def curvelength(x):
    cl = 0
    for i in range(x.shape[0]-1):
        cl += abs(x[i]-x[i+1])
    return cl


def energy(x):
    return np.sum(np.multiply(x, x))


def nonlinear_energy(x):
    # NLE(x[n]) = x**2[n] - x[n+1]*x[n-1]
    x_squared = x[1:-1]**2
    subtrahend = x[2:]*x[:-2]
    return np.sum(x_squared-subtrahend)


# Enhancement feature not used, Clayton, 2019 uses it though
# def ehf(x, prev):
#     # (based on Basar et. al. 1983)
#     # "prev" is array of values from prior context
#     rms = np.sqrt(np.mean(prev**2))
#     return 2*np.sqrt(2)*(max(x)/rms)


def spec_entropy(x):
    return entropy.spectral_entropy(x, fs, method="welch", normalize=True)


def integral(x):
    return integrate.simps(x)


def stddeviation(x):
    return np.std(x)


def variance(x):
    return np.var(x)


def skew(x):
    return stats.skew(x)


def kurtosis(x):
    return stats.kurtosis(x)

# added ones

# some of these are nicked from https://github.com/raphaelvallat/entropy


def sample_entropy(x):
    return entropy.sample_entropy(x, order=2, metric='chebyshev')


def perm_entropy(x):
    return entropy.perm_entropy(x, order=3, normalize=True)


def svd_entropy(x):
    return entropy.svd_entropy(x, order=3, delay=1, normalize=True)


def app_entropy(x):
    return entropy.app_entropy(x, order=2, metric='chebyshev')


def petrosian(x):
    return entropy.petrosian_fd(x)


def katz(x):
    return entropy.katz_fd(x)


def higuchi(x):
    return entropy.higuchi_fd(x, kmax=10)


def rootmeansquare(x):
    return np.sqrt(np.mean(x**2))


def dfa(x):
    return entropy.detrended_fluctuation(x)


funclist = [mean, absmean, maximum, absmax, minimum, absmin, minplusmax, maxminusmin, curvelength, energy, nonlinear_energy, integral, stddeviation,
            variance, skew, kurtosis, np.sum, spec_entropy, sample_entropy, perm_entropy, svd_entropy, app_entropy, petrosian, katz, higuchi, rootmeansquare, dfa]


def window_data(data: np.ndarray):
    """windows the data
    (using a stride length of 1)
    """

    w_len = 128
    stride = w_len // 2

    no_offset_windows = np.split(data, 10)
    offset_windows = np.split(data[stride:-stride], 9)
    windows = [0] * 19
    windows[::2] = no_offset_windows
    windows[1::2] = offset_windows
    windows = np.array(windows, dtype=np.float32)

    return windows


def feats_array_4_window(window: np.ndarray):
    """Takes a single window, returns an array of features of 
    shape (n.features, electrodes), and then flattens it 
    into a vector
    """

    outvec = np.zeros((len(funclist), window.shape[1]))

    for i in range(len(funclist)):
        for j in range(window.shape[1]):
            outvec[i, j] = funclist[i](window[:, j])

    outvec = outvec.reshape(-1)

    return outvec


def make_simple_feats(windowed_data: np.ndarray):

    simple_feats = []

    for w in range(len(windowed_data)):
        simple_feats.append(feats_array_4_window(windowed_data[w]))

    return(np.array(simple_feats))


def add_deltas(feats_array: np.ndarray):

    deltas = np.diff(feats_array, axis=0)
    double_deltas = np.diff(deltas, axis=0)
    all_feats = np.hstack((feats_array[2:], deltas[1:], double_deltas))

    return(all_feats)


def make_features_per_epoch(epoch):
    epoch = window_data(epoch)
    epoch = make_simple_feats(epoch)
    epoch = add_deltas(epoch)
    return(epoch)


def get_feats_list():
    feats_list = []
    feats_list += [func.__name__ for func in funclist]
    feats_list += ["d_" + func.__name__ for func in funclist]
    feats_list += ["dd_" + func.__name__ for func in funclist]
    return(feats_list)


def get_labels(experiments_dir, experiment, svm_features_dir):

    print("Saving labels for experiment {0}".format(experiment))

    if not op.exists(op.join(svm_features_dir, experiment)):
        os.mkdir(op.join(svm_features_dir, experiment))

    numpy_features = np.genfromtxt(
        op.join(experiments_dir, experiment, "speaking.csv"), delimiter=",", dtype=str)
    eeg_labels = numpy_features[1::1280, 16]
    np.save(op.join(svm_features_dir, experiment, "labels.npy"), eeg_labels)


def bandpass(raw_eeg, low=1, high=None):
    """creates an MNE filter object 

    Args:
        raw_eeg: to-be-filtered data object
        low: lowpass boundary (Hz) - defaults to 1Hz
        high: highpass boundary (Hz) - defaults to 128Hz

    Returns:
        filtered_eeg._data: numpy ndarray with the bandpassed data
    """
    CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7',
                'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

    # Sampling rate
    sfreq = 256  # Hz

    raw_eeg = raw_eeg.transpose()

    # Create the info structure needed by MNE
    info = mne.create_info(ch_names=CHANNELS, sfreq=sfreq, ch_types='eeg')
    # create the Raw object
    raw = mne.io.RawArray(raw_eeg, info)

    filtered_eeg = raw.filter(l_freq=low, h_freq=None, picks=None, filter_length='auto',
                              l_trans_bandwidth='auto', h_trans_bandwidth='auto', fir_design='firwin')

    return filtered_eeg._data.transpose()


def make_features(experiments_dir, experiment, epoch_type, svm_features_dir):
    print("Making SVM features for {0} epoch, experiment {1}".format(
        epoch_type, experiment))
    if not op.exists(op.join(svm_features_dir, experiment)):
        os.mkdir(op.join(svm_features_dir, experiment))

    numpy_features = np.genfromtxt(op.join(
        experiments_dir, experiment, epoch_type + ".csv"), delimiter=",", dtype=float)
    raw_eeg = numpy_features[1:, 2:16].astype(np.float32)

    # ==============================================================================
    # this is an easy function for bandpassing
    raw_eeg = bandpass(raw_eeg, low=1)
    # ==============================================================================

    # Our sampling frequency is 256, each token has five seconds of EEG data
    n_tokens = len(raw_eeg)/256/5
    if not n_tokens % 1 == 0:
        raise TypeError("'{0} features' from experiment {1} doesn't seem to contain the right number of samples \n \
        Number of samples should be (n_prompts * sampling frequency (256) * token length(5s)) \n \
        Sample length recieved is {2}.".format(epoch_type, experiment, len(raw_eeg)))  # deleted n_tokens from the fourth place in .format()

    raw_eeg = np.split(raw_eeg, n_tokens)

    epochs = []
    for i, epoch in enumerate(raw_eeg):
        print("Making features for token {0} of {1}, epoch type '{2}', experiment {3}".format(
            i+1, len(raw_eeg), epoch_type, experiment))
        epoch = make_features_per_epoch(epoch)
        epochs.append(epoch)

    epochs = np.array(epochs, dtype=np.float32)
    np.save(op.join(svm_features_dir, experiment, epoch_type), epochs)


if __name__ == "__main__":

    if not op.exists("../experiments/svm_features"):
        os.mkdir("../experiments/svm_features")
    svm_features_dir = "../experiments/svm_features"

    experiments_dir = "../experiments/"
    experiments_list = sorted(os.listdir(experiments_dir))

    print("using the following features:")
    print(get_feats_list())

    for experiment in experiments_list:
        get_labels(experiments_dir, experiment, svm_features_dir)
        for epoch_type in ["stimuli", "thinking", "speaking"]:  # stimuli = hearing
            make_features(experiments_dir, experiment,
                          epoch_type, svm_features_dir)
