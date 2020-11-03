from mne.preprocessing import ICA, create_eog_epochs, corrmap
from EEG_data import EEG_data
import numpy as np
import mne
import matplotlib.pyplot as plt
import os

# most of this is based on an MNE tutorial, which can be found here:
# https: // mne.tools/0.17/auto_tutorials/plot_artifacts_correction_ica.html


class EEG_ICA():
    """
    class for easy ICA application to the FEIS data
    """

    def __init__(self, EEG_data):
        """initializes EEG_ICA object

        Args:
           EEG_data: EEG_data object containing the data from a csv file
        """
        print("initializing ICA")
        self.sfreq = EEG_data.sfreq  # Hz
        self.subject = EEG_data.subject
        self.category = EEG_data.category
        self.full_data = EEG_data.full_data
        self.data = EEG_data.data
        self.labels = EEG_data.labels_numerical

        # Create the info structure needed by MNE
        self._info = mne.create_info(ch_names=EEG_data.columns[:-1],
                                     sfreq=self.sfreq, ch_types='eeg')

        # create the Raw object

        self.raw = mne.io.RawArray(
            self.data.transpose(), self._info)
        print(
            f"data loaded: participant {self.subject}, category {self.category}")

    def set_eog_channels(self, channels=["AF3", "AF4"]):
        """sets specified channels as eog

        Args:
            channels (list of strings, optional): channels that contain ocular information. Defaults to ["AF3", "AF4"].
        """
        for channel in channels:
            self.filtered.set_channel_types({str(channel): 'eog'})
        return self.filtered

    def reset_eog_channels(self, channels=["AF3", "AF4"]):
        """sets specified channels as eog

        Args:
            channels (list of strings, optional): channels that contain ocular information. Defaults to ["AF3", "AF4"].
        """
        for channel in channels:
            self.filtered.set_channel_types({str(channel): 'eeg'})
        return self.filtered

    def apply_filter(self, l_freq=1, h_freq=None, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', fir_design='firwin'):
        """applies a specified filter to the data before fitting the ICA
        Generally, a 1Hz low-pass filter is recommended

        Args:
            l_freq (int, optional): low_pass boundary. Defaults to 1.
            h_freq (int, optional): high-pass boundary. Defaults to None.
            filter_length (int, optional): filter length. Defaults to 'auto'.
            l_trans_bandwidth (int, optional): low-pass bandwith. Defaults to 'auto'.
            h_trans_bandwidth (int, optional): high-pass bandwidth. Defaults to 'auto'.
            fir_design (str, optional): filter design. Defaults to 'firwin'.
        """
        self.filtered = self.raw.filter(l_freq=l_freq, h_freq=h_freq, picks=None, filter_length=filter_length,
                                        l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, fir_design=fir_design)
        return self.filtered

    def apply_ICA(self, method="fastica", n_components=14, decim=3, random_state=None, exclude=[]):
        """applies the specified ICA method

        Args:
            method (str, optional): various ICA methods available in the MNE package. Defaults to "fastica".
            n_components (int, optional): number of components. Defaults to 14 (number of channels).
            decim (int, optional): number of decimals for degree of accuracy. Defaults to 3.
            random_state (int, optional): random seed for repeatability. Defaults to None.
            exclude (list, optional): option to exclude certain channel types. Defaults to [].
        """

        self._picks_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=False,
                                         stim=False, exclude=exclude)

        ten_twenty_montage = mne.channels.make_standard_montage(
            'standard_1020')
        self.filtered.set_montage(ten_twenty_montage)

        # creating and fitting the ICA on the filtered data
        self._ica = ICA(n_components=n_components,
                        method=method, random_state=random_state)
        self._ica.fit(self.filtered, picks="all",
                      decim=decim, reject=None)

    def find_eog_artifacts(self):
        self._eog_average = create_eog_epochs(
            self.filtered, picks=self._picks_eeg).average()
        self._eog_epochs = create_eog_epochs(
            self.filtered)  # get single EOG trials
        self._eog_inds, self._eog_scores = self._ica.find_bads_eog(
            self._eog_epochs)  # find via correlation

    def remove_eog_artifacts(self):
        self._ica.exclude.extend(self._eog_inds)


class EEG_ICA_resample():
    """
    class for easy ICA application to the FEIS data during resampling with the pandas concat function
    """

    def __init__(self, data):
        """initializes EEG_ICA object

        Args:
           EEG_data: EEG_data object containing the data from a csv file
        """
        print("initializing ICA")
        self.sfreq = 256  # Hz
        self.data = data

        # Create the info structure needed by MNE
        self._info = mne.create_info(ch_names=list(self.data.columns),
                                     sfreq=self.sfreq, ch_types='eeg')

        # create the Raw object
        self.raw = mne.io.RawArray(
            self.data.transpose(), self._info)
        print(
            f"data loaded for the ICA")

    def set_eog_channels(self, channels=["AF3", "AF4"]):
        """sets specified channels as eog

        Args:
            channels (list of strings, optional): channels that contain ocular information. Defaults to ["AF3", "AF4"].
        """
        for channel in channels:
            self.filtered.set_channel_types({str(channel): 'eog'})
        return self.filtered

    def reset_eog_channels(self, channels=["AF3", "AF4"]):
        """sets specified channels as eog

        Args:
            channels (list of strings, optional): channels that contain ocular information. Defaults to ["AF3", "AF4"].
        """
        for channel in channels:
            self.filtered.set_channel_types({str(channel): 'eeg'})
        return self.filtered

    def apply_filter(self, l_freq=1, h_freq=None, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', fir_design='firwin'):
        """applies a specified filter to the data before fitting the ICA
        Generally, a 1Hz low-pass filter is recommended

        Args:
            l_freq (int, optional): low_pass boundary. Defaults to 1.
            h_freq (int, optional): high-pass boundary. Defaults to None.
            filter_length (int, optional): filter length. Defaults to 'auto'.
            l_trans_bandwidth (int, optional): low-pass bandwith. Defaults to 'auto'.
            h_trans_bandwidth (int, optional): high-pass bandwidth. Defaults to 'auto'.
            fir_design (str, optional): filter design. Defaults to 'firwin'.
        """
        self.filtered = self.raw.filter(l_freq=l_freq, h_freq=h_freq, picks=None, filter_length=filter_length,
                                        l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, fir_design=fir_design)
        return self.filtered

    def apply_ICA(self, method="fastica", n_components=14, decim=3, random_state=None, exclude=[]):
        """applies the specified ICA method

        Args:
            method (str, optional): various ICA methods available in the MNE package. Defaults to "fastica".
            n_components (int, optional): number of components. Defaults to 14 (number of channels).
            decim (int, optional): number of decimals for degree of accuracy. Defaults to 3.
            random_state (int, optional): random seed for repeatability. Defaults to None.
            exclude (list, optional): option to exclude certain channel types. Defaults to [].
        """

        self._picks_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=False,
                                         stim=False, exclude=exclude)

        ten_twenty_montage = mne.channels.make_standard_montage(
            'standard_1020')
        self.filtered.set_montage(ten_twenty_montage)

        # creating and fitting the ICA on the filtered data
        self._ica = ICA(n_components=n_components,
                        method=method, random_state=random_state)
        self._ica.fit(self.filtered, picks="all",
                      decim=decim, reject=None)

    def find_eog_artifacts(self):
        self._eog_average = create_eog_epochs(
            self.filtered, picks=self._picks_eeg).average()
        self._eog_epochs = create_eog_epochs(
            self.filtered)  # get single EOG trials
        self._eog_inds, self._eog_scores = self._ica.find_bads_eog(
            self._eog_epochs)  # find via correlation

    def remove_eog_artifacts(self):
        self._ica.exclude.extend(self._eog_inds)
