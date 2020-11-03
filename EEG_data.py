import matplotlib.pyplot as plt
from matplotlib import rcParams
import itertools
import numpy as np
import zipfile
import os
import shutil
from pprint import pprint
import pandas as pd
from torch import Tensor
import torch
import mne


class EEG_data:
    """
    Class for EEG data objects.
    """

    def __init__(self, subject, category="full_eeg", columns=None, sfreq=256):
        """[summary]

        Args:
            subject (int): participant_id, between (and including) 1 and 21
            category (string): specify one or more of {articulators, full_eeg, resting, speaking, stimuli, thinking}. Defaults to full_eeg.

        Raises:
            SyntaxError: if specified subject is not between 1 and 21 (chinese participants not currently supported),
            specified category is not available, or the data type of the input is not accepted
        """
        # participants 1-9 are called 01-09

        if subject not in range(1, 22) or not isinstance(subject, int):
            raise SyntaxError(
                "please check your subject input to your EEG_data object")
        if subject < 10:
            subject = f'0{subject}'
        if str(category).lower() not in {"articulators", "full_eeg", "resting", "speaking", "stimuli", "thinking"} or not isinstance(category, str):
            raise SyntaxError(
                "please check your category input to your EEG_data object")

        self.subject = subject
        self.category = category.lower()
        self.data_folder = f"./FEIS/data_eeg/{subject}"
        self.data_csv = os.path.join(
            os.getcwd(), self.data_folder, self.category + ".csv")
        self.data = None
        self.data_loaded = False
        if columns == "channels":
            self.columns = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7',
                            'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        elif columns == "channels+labels":
            self.columns = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7',
                            'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'Label']
        else:
            self.colnames = columns
        self.data_tensor = None
        self.sfreq = sfreq

    def get_dataframe(self):
        return pd.DataFrame(self.data)

    def load_data(self):
        """
        loads specified csv data. Unzips the .zip file first, if necessary.

        loads self.data attribute
        sets self.data_loaded attribute to True
        loads self.colnames attribute
        """
        try:
            if self.columns == None:
                data = pd.read_csv(self.data_folder +
                                   f"/{self.category}.csv", delimiter=",")
            else:
                data = pd.read_csv(self.data_folder +
                                   f"/{self.category}.csv", delimiter=",", usecols=self.columns)

        except FileNotFoundError:  # raised if there is no csv file with the specified name
            self.unzip()
            # try again
            if self.columns == None:
                data = pd.read_csv(self.data_folder +
                                   f"/{self.category}.csv", delimiter=",")
            else:
                data = pd.read_csv(self.data_folder +
                                   f"/{self.category}.csv", delimiter=",", usecols=self.columns)

        self.full_data = data
        self.data = data.drop("Label", axis=1)
        _v1 = data["Label"] == "goose"
        _v2 = data["Label"] == "thought"
        _v3 = data["Label"] == "fleece"
        _v4 = data["Label"] == "trap"
        self.data_vowels = data[_v1 | _v2 | _v3 | _v4]
        self.data_consonants = data[~data.isin(self.data_vowels)].dropna()
        self.labels = data["Label"]
        self.labels_numerical = self.recode_labels()
        self.data_loaded = True
        self.colnames = self.data.columns
        return self.data

    def recode_labels(self):
        self.label_mapping = {label: i for i,
                              label in enumerate(self.labels.unique())}
        _labels_numerical = [self.label_mapping[item] for item in self.labels]
        return _labels_numerical

    def unzip(self):
        """
        unzips an archive if requested but not unzipped yet
        """
        with zipfile.ZipFile(self.data_folder + f"/{self.category}.zip", 'r') as zipped_dir:
            zipped_dir.extractall(self.data_folder)

    def move_to_experiments(self):
        exp_dir = os.path.join("./experiments")

        if not os.path.isdir(os.path.join(exp_dir, f"Ex_P{self.subject}")):
            os.mkdir(os.path.join(exp_dir, f"Ex_P{self.subject}"))
        if not os.path.isfile(os.path.join(exp_dir, f"Ex_P{self.subject}/{self.category}.csv")):
            shutil.copy2(os.path.join(
                self.data_folder, f"{self.category}.csv"), os.path.join(exp_dir, f"Ex_P{self.subject}/{self.category}.csv"))

    def print_data(self):
        pprint(self.data)

    def print_attributes(self):
        attributes = {"subject": self.subject,
                      "category": self.category,
                      "data_folder": self.data_folder,
                      "data_csv": self.data_csv,
                      "data_loaded": self.data_loaded,
                      "column names": self.colnames}
        pprint(attributes)

    def as_tensor(self, dataframe=None):
        """returns a tensor version of the data

        Returns:
            torch.Tensor: data in pytorch tensor format
        """
        if dataframe:
            self.data_tensor = Tensor(dataframe)
        else:
            self.data_tensor = Tensor(
                self.data._data.as_array())
        return self.data_tensor


# Example usage EEG_data
# NB: only the thinking, speaking and stimuli categories will be needed for SVM feature extraction

# Individual actions:
# stimuli_eeg_19 = EEG_data(subject=19, category="stimuli")
# stimuli_eeg_19.load_data()
# stimuli_eeg_19.move_to_experiments()
# stimuli_eeg_19.print_attributes()
# stimuli_eeg_19.print_data()

# for all English data
# for subject in range(1, 22):
#     for category in ["thinking", "stimuli", "speaking"]:
#         temp_ = EEG_data(subject=subject, category=category)
#         temp_.load_data()
#         temp_.move_to_experiments()
#         print(
#             f"finished loading and copying subject {subject}'s {category} data'")
