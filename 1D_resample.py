from ICA import EEG_ICA_resample
from EEG_data import EEG_data
from utils import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import shutil
import os
import argparse
import mne
import math

torch.set_printoptions(linewidth=120)  # better printing readability


class CNN(nn.Module):
    def __init__(self, lrate, binary, batch_size=1, save=True):
        """initialize CNN object

        Args:
            lrate (float): learning rate
            binary (bool): whether the output layer should have a binary class
            batch_size (int, optional): Defaults to 8.
            save (bool, optional): whether to save checkpoints for latest and best iteration. Defaults to True.
        """
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.lrate = lrate
        self.device = self.get_device()
        # convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=14, out_channels=128, kernel_size=3, stride=1, bias=True)
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True)
        self.conv4 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True)
        # linear layers
        self.fc1 = nn.Linear(in_features=128*1272,
                             out_features=100, bias=True)
        self.fc2 = nn.Linear(
            in_features=100, out_features=50, bias=True)
        # output layer, depending on binary boolean
        if binary:
            self.out = nn.Linear(
                in_features=128*1276, out_features=2, bias=True)
        else:
            self.out = nn.Linear(in_features=50, out_features=16,
                                 bias=True)
        self.dropoutlayer = nn.Dropout(p=0.5)

        # initializing tracking variables for the best results
        self.best_loss = math.inf
        self.best_epoch = 0

        # creating the save directory
        if save:
            self.create_save_dir()
            self.save = True
        else:
            self.save = False

    def get_device(self):
        """determines best available computing device

        Returns:
            device: torch device
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device

    def forward(self, t):
        """forward function of the network

        Args:
            t (torch.Tensor): data without labels

        Returns:
            t (torch.Tensor): transformed data tensor
        """
        # NB: input layer is implicit: t=t

        # first convolutional layer

        # print(f"input t shape : {t.shape}")
        t = self.conv1(t)
        t = F.relu(t)
        # print(f"t shape after first layer {t.shape}")

        # second convolutional layer
        t = self.conv2(t)
        # print(f"t shape after second layer {t.shape}")
        t = F.relu(t)

        # # third convolutional layer
        # t = self.conv3(t)
        # # print(f"t shape after third layer {t.shape}")
        # t = F.relu(t)

        # # fourth convolutional layer
        # t = self.conv4(t)
        # # print(f"t shape after fourth layer {t.shape}")
        # t = F.relu(t)

        t = t.view(t.shape[0], t.shape[1] * t.shape[2])

        # # first hidden linear layer
        # t = self.fc1(t)
        # t = F.relu(t)
        # t = self.dropoutlayer(t)
        # # print(f"t shape after first linear layer: {t.shape}")

        # # second hidden linear layer
        # t = self.fc2(t)
        # t = F.relu(t)
        # t = self.dropoutlayer(t)
        # # print(f"t shape after second linear layer: {t.shape}")

        # output layer
        t = self.out(t)
        # print(f"t shape after output linear layer: {t.shape}")
        # NB: softmax not necessary since nn.CrossEntropyLoss criterion does that

        return t

    def load_external_training_data(self, data):
        """loads external training data

        Args:
            data (torch.Tensor): data with labels
        """
        self._train_set = data.to(self.device)
        self.train_loader = torch.utils.data.DataLoader(
            self._train_set, batch_size=self.batch_size)
        print("training data loaded")

    def load_external_validation_data(self, data):
        """loads external validation data

        Args:
            data (torch.Tensor): data with labels
        """
        self._val_set = data.to(self.device)
        self.val_loader = torch.utils.data.DataLoader(
            self._val_set, batch_size=self.batch_size)
        print("validation data loaded")

    def load_external_test_data(self, data):
        """loads external test data

        Args:
            data (torch.Tensor): data with labels
        """
        self._test_set = data.to(self.device)
        self.test_loader = torch.utils.data.DataLoader(
            self._test_set, batch_size=100000)
        print("test data loaded")

    def train_model(self, max_epochs, model, criterion):
        """trains the CNN using all predefined settings

        Args:
            max_epochs (int): maximum number of epochs allowed
            model (CNN object): model instance
            criterion (torch lossfunction): generally nn.CrossEntropyLoss

        Returns:
            model (CNN object): trained model
        """
        # initialize tracking variables for early convergence
        prev_bl = math.inf
        identical_loss = 0

        for epoch in range(args.max_epochs):
            ############
            # training #
            ############
            train_loss = 0
            total_correct_train = 0
            model.train()
            for batch in model.train_loader:
                data, labels = torch.split(
                    batch.float(), [14, 1], dim=-2)

                preds = model(data)  # passing batch through the network

                labels = torch.mean(torch.mean(
                    labels.float(), dim=1), dim=1)

                # Calculating the loss
                training_loss = criterion(preds, labels.round().long())
                train_loss += training_loss.item()

                total_correct_train += model.get_num_correct(preds, labels)

                # zeroing out the gradients
                optimizer.zero_grad()
                # necessary to prevent gradients from accumulating

                training_loss.backward()  # calculate gradients
                optimizer.step()  # update weights

            ##############
            # validating #
            ##############
            model.eval()  # putting model in evaluation mode, not updating gradients anymore
            valid_loss = 0
            total_val_correct = 0
            for batch in model.val_loader:
                data, labels = torch.split(
                    batch.float(), [14, 1], dim=-2)

                preds = model(data)

                labels = torch.mean(torch.mean(
                    labels.float(), dim=1), dim=1)

                # Calculating the loss
                validation_loss = criterion(preds, labels.round().long())
                valid_loss += validation_loss.item()

                total_val_correct += model.get_num_correct(preds, labels)

            if valid_loss < model.best_loss:
                # updating the best parameters
                model.best_loss = valid_loss
                model.best_epoch = epoch
                if model.save:
                    model.save_checkpoint(
                        epoch, valid_loss, optimizer, is_best=True)
            elif valid_loss == prev_bl:
                if identical_loss < 10:
                    identical_loss += 1
                else:
                    print(
                        f"converged early at epoch {epoch+1}, validation loss: {valid_loss}")
                    break
                if model.save:
                    model.save_checkpoint(epoch, valid_loss,
                                          optimizer, is_best=False)
            elif valid_loss != prev_bl:
                identical_loss = 0
                if model.save:
                    model.save_checkpoint(
                        epoch, valid_loss, optimizer, is_best=False)
            prev_bl = valid_loss
            print(
                f"\nepoch: {epoch+1}\ntotal_val_correct: {total_val_correct}/{len(model._val_set)}, total_val_loss: {valid_loss}")
            print(
                f"total_train_correct: {total_correct_train}/{len(model._train_set)}, total_train_loss: {train_loss}")

        return model

    def create_save_dir(self):
        """
        creates saving directory named with current timestamp and sets self.save_path
        """

        if not os.path.isdir("./experiments"):
            if not os.path.isdir("./TEMP_save"):
                os.mkdir("./TEMP_save/")
            self._sv_pth = "./TEMP_save/"
            print(f"\nWARNING: save location = {os.getcwd() + '/TEMP_save'}")
        else:
            if not os.path.isdir("./experiments/CNN_saves"):
                os.mkdir("./experiments/CNN_saves")
            self._sv_pth = "./experiments/CNN_saves"

        now = dt.datetime.now()
        sv_folder = now.strftime("%m%d%H%M%S")

        self.save_path = os.path.join(self._sv_pth, sv_folder)

        os.mkdir(self.save_path)
        os.mkdir(os.path.join(self.save_path, "latest_checkpoint"))
        os.mkdir(os.path.join(self.save_path, "best_checkpoint"))

    def save_checkpoint(self, epoch, valid_loss, optimizer, is_best):
        """saves checkpoints of CNN states

        Args:
            epoch (int): checkpoint epoch
            valid_loss (float): validation loss
            optimizer (torch optimizer): optimizer used
            is_best (bool): whether checkpoint should be copied to the best_checkpoint folder
        """

        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'save_folder': self.save_path
        }

        torch.save(checkpoint, self.save_path+"/latest_checkpoint/checkpoint")

        if is_best:
            shutil.copyfile(self.save_path+"/latest_checkpoint/checkpoint",
                            self.save_path+"/best_checkpoint/checkpoint")

    def load_checkpoint(self, optimizer, cp, folder="best"):
        """loads a checkpoint from earlier training

        Args:
            optimizer (torch optimizer): untrained optimizer
            cp (string): checkpoints folder path
            folder (str, optional): {'best' or 'latest'}. Defaults to "best".

        Returns:
            model, (CNN object): CNN
            optimizer (torch optimizer): trained optimizer
            epoch (int): epoch training ended
            valid_loss (float): current validation loss
        """
        if folder == "best":
            path = f"{self._sv_pth}{cp}/best_checkpoint/checkpoint"
        elif folder == "latest":
            path = f"{self._sv_pth}{cp}/latest_checkpoint/checkpoint"
        else:
            print("please check your folder input in the load_checkpoint function")

        chckpnt = torch.load(path)
        self.load_state_dict(chckpnt["state_dict"])
        optimizer.load_state_dict(chckpnt["optimizer"])
        validation_loss = chckpnt["valid_loss_min"]

        return self, optimizer, chckpnt["epoch"], validation_loss

    def print_params_to_save(self):
        """
        prints CNN object settings to txt file
        """
        print(
            f"\nsaving network parameters at {self.save_path}/used_settings.txt")
        settings = self.get_all_settings()
        parameters = self.get_all_named_parameter_shapes()
        with open(f"{self.save_path}/used_settings.txt", "w+") as file:
            # list comprension method inspired by DavideL at
            # https://stackoverflow.com/questions/36965507/writing-a-dictionary-to-a-text-file
            lst_set = [f'{k} : {settings[k]}' for k in settings]
            lst_par = [f'{k} : {parameters[k]}' for k in parameters]
            file.write("Settings:\n")
            [file.write(f'{s}\n') for s in lst_set]
            file.write("\nNamed Parameter Shapes:\n")
            [file.write(f'{s}\n') for s in lst_par]

    def get_model_accuracy(self, model):
        """[summary]

        Args:
            model (CNN object): trained CNN

        Returns:
            accuracy (float): accuracy between 0 and 1
        """
        for batch in self.test_loader:
            data, labels = torch.split(
                batch.float(), [14, 1], dim=-2)

            preds = model(data)
            correct = self.get_num_correct(
                preds, labels)

            return correct/len(labels)

    def get_all_settings(self):
        """returns all settings used

        Returns:
            settings [dict]: k:v of settings used
        """
        return {
            "participant": args.participant,
            "category": args.category,
            "lrate": args.learning_rate,
            "batch size": self.batch_size,
            "max_epochs": args.max_epochs,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "save": self.save
        }

    def get_all_named_parameter_shapes(self):
        """returns all named parameters in the current model
        """
        return {name: param.shape for name, param in self.named_parameters()}

    def get_num_correct(self, preds, labels):
        """returns number of correct predictions

        Args:
            preds (torch.Tensor): CNN model output, predicted labels
            labels (torch.Tensor): targets for every data instance

        Returns:
            (int): number of correct predictions
        """
        preds = torch.argmax(preds, dim=1)
        if labels.shape != torch.Size([len(labels)]):
            labels = torch.mean(torch.mean(labels, dim=1), dim=1)
        return preds.eq(labels).sum().item()

    @ torch.no_grad()  # not using gradient tracking to preserve memory
    def plot_confusion_matrix(self, label_mapping, category, show=False, title="Confusion Matrix"):
        """generates confusion matrix

        Args:
            label_mapping (dict): numerical to string label mapping
            show (bool, optional): whether to show the plot. Defaults to False.
            category (str, optional): one of {'training', 'validation', 'testing'}. Defaults to "training".
        """
        # method adapted from: https://deeplizard.com/learn/video/0LhiS6yu2qQ
        if category == "training":
            _set = self._train_set
        elif category == "validation":
            _set = self._val_set
        elif category == "testing":
            _set = self._test_set

        # rapidly predicting all labels
        prediction_loader = torch.utils.data.DataLoader(
            _set, batch_size=len(_set))
        preds, labels = self._get_all_preds(self, prediction_loader, _set)
        preds_correct = preds.eq(labels).sum().item()

        print(f"\nprediction tensor created with {preds.shape}\n")
        print(f"total correct during {category}: {preds_correct}/{len(_set)}")
        print(f"{category} accuarcy: {preds_correct/len(_set)}")

        # plotting the matrix
        cm = confusion_matrix(
            labels, preds)
        names = label_mapping.keys()
        plt.figure(figsize=(len(names)*2.5, len(names)*2.5))
        plot_confusion_matrix(cm, names, title=title)

        if self.save:
            if os.path.exists(self.save_path+f"/conf_matrix_{category}.svg"):
                plt.savefig(fname=self.save_path +
                            f"/conf_matrix_{category}_1.svg")
            else:
                plt.savefig(fname=self.save_path +
                            f"/conf_matrix_{category}.svg")
        if show:
            plt.show()

    def _get_all_preds(self, model, loader, _set):
        """gets predictions for all data

        Args:
            model (CNN object): trained model
            loader (dataloader object): dataloader object

        Returns:
            all_preds: all predicted labels
            lables: actual labels
        """
        all_preds = torch.Tensor([])
        all_labels = torch.Tensor([])
        for batch in loader:
            data, labels = torch.split(
                batch.float(), [14, 1], dim=-2)
            preds = model(data)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels))

        preds = torch.argmax(all_preds.squeeze(), dim=1)
        labels = torch.mean(all_labels.view(
            len(_set), -1), dim=1).round().long()

        return preds, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CNN classifier for FEIS data")
    mode = parser.add_mutually_exclusive_group()
    classification = parser.add_mutually_exclusive_group()
    mode.add_argument("-t", "--train", action="store_true",
                      help="train CNN")
    mode.add_argument("-d", "--decode", action="store_true",
                      help="test CNN on test batch")
    classification.add_argument("-bc", "--binary_class", action="store_true",
                                help="divide the data in binary vowel/consonant classes")
    classification.add_argument("-bv", "--binary_voicing", action="store_true",
                                help="divide the data in binary vowel/consonant classes")
    classification.add_argument("-ph", "--phonemes", action="store_true",
                                help="no binary encoding of the data: classification based on the phonemes")
    parser.add_argument("-p", "--participant", type=int, default=19,
                        help="which subjects data to use")
    parser.add_argument("-c", "--category", type=str, default="speaking",
                        help="which category of the subjects data to use")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.001, help="learning rate for training")
    parser.add_argument("-bs", "--batch_size", type=int,
                        default=8, help="batch size during training")
    parser.add_argument("-me", "--max_epochs", type=int,
                        default=100, help="maximum training epochs")
    parser.add_argument("-l", "--load", type=str,
                        help="the directory in CNN_saves you want to load a checkpoint from")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save checkpoints of trained model")
    args = parser.parse_args()

    # setting random seed for repeatibility
    random_seed = 14
    print(f"random seed: {random_seed}")
    torch.manual_seed(random_seed)
    print(f"category {args.category}")

    # initializing and loading the data and labels
    EEG_data = EEG_data(subject=args.participant, category=args.category,
                        columns="channels+labels")
    EEG_data.load_data()
    data_vowels = EEG_data.data_vowels
    data_consonants = EEG_data.data_consonants
    if args.binary_class:
        full_data = pd.concat([data_consonants, data_vowels,
                               data_vowels, data_vowels], ignore_index=True)
    elif args.binary_voicing:
        full_data = pd.concat([], ignore_index=True)
    label_mapping = {label: i for i,
                     label in enumerate(full_data["Label"].unique())}
    targets = [label_mapping[item] for item in full_data["Label"]]
    data = full_data.drop("Label", axis=1)

    # Recoding the labels for the specific task
    if args.binary_class:
        # recoding the labels into binary vowel/consonants
        # goose, thought, fleece, trap in label_mapping
        vowels = [12, 13, 14, 15]
        targets = [1 if label in vowels else 0 for label in targets]
        label_mapping = {"consonant": 0, "vowel": 1}
    elif args.binary_voicing:
        # recoding the labels into binary vowel/consonants
        # goose, thought, zh, n, fleese, trap, ng, z, m, v in label_mapping
        voiced = [0, 1, 2, 5, 7, 8, 10, 13, 14, 15]
        targets = [1 if label in voiced else 0 for label in targets]
        label_mapping = {"unvoiced": 0, "voiced": 1}

    # Preprocessing the data with an ICA
    ICA = EEG_ICA_resample(data=data)
    ICA.apply_filter()  # highpassing at 1Hz
    ICA.set_eog_channels()  # defaults are the EPOC+ front two channels
    ICA.apply_ICA()
    ICA.find_eog_artifacts()
    ICA.remove_eog_artifacts()  # auto removing found eog artifacts
    ICA.reset_eog_channels()  # merging the makeshift eog channels back in

    # Initializing the CNN
    if args.save:
        save = True
    else:
        save = False
    if args.binary_class or args.binary_voicing:
        CNN = CNN(batch_size=args.batch_size,
                  lrate=args.learning_rate, binary=True, save=save)
    elif args.phonemes:
        CNN = CNN(batch_size=args.batch_size,
                  lrate=args.learning_rate, binary=False, save=save)
    CNN = CNN.to(CNN.device)

    # intializing weights for loss function
    if args.binary_class:
        weight = torch.Tensor(
            [8/16, 8/16]
        ).to(CNN.device)
    elif args.binary_voicing:
        weight = torch.Tensor(
            [10/16, 6/16]
        ).to(CNN.device)
    else:
        weight = torch.Tensor(
            [1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16,
             1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16]
        ).to(CNN.device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(CNN.parameters(), lr=args.learning_rate)

    # converting the preprocessed data into tensors
    train_tensor = torch.from_numpy(
        ICA.filtered._data).reshape(240, 14, -1).double().to(CNN.device)
    targets_tensor = torch.Tensor(
        targets).double().unsqueeze(dim=1).to(CNN.device)

    # Splitting into the training/test sets
    targets_tensor = targets_tensor.reshape(240, 1, -1).to(CNN.device)
    targets = torch.mean(targets_tensor, dim=1).squeeze()
    _data, test_data, _labels, test_labels = train_test_split(
        train_tensor, targets, test_size=0.2, random_state=random_seed)
    train_data, val_data, train_labels, val_labels = train_test_split(
        _data, _labels, test_size=0.1875, random_state=random_seed)

    train_labels = train_labels.unsqueeze(dim=1)
    test_labels = test_labels.unsqueeze(dim=1)
    val_labels = val_labels.unsqueeze(dim=1)

    # combining the data and label tensors
    full_train_tensor = torch.cat(
        (train_data, train_labels), dim=1).to(CNN.device)
    full_val_tensor = torch.cat(
        (val_data, val_labels), dim=1).to(CNN.device)
    full_test_tensor = torch.cat(
        (test_data, test_labels), dim=1).to(CNN.device)

    print(f"\ntrain data shape: {full_train_tensor.shape}")
    print(f"validation data shape: {full_val_tensor.shape}")
    print(f"test data shape: {full_test_tensor.shape}\n")

    if args.train:
        CNN.load_external_training_data(data=full_train_tensor)
        CNN.load_external_validation_data(data=full_val_tensor)
        print("started training...\n")

        CNN = CNN.train_model(
            max_epochs=args.max_epochs, model=CNN, criterion=criterion)

        if args.save:
            CNN.print_params_to_save()
        CNN.plot_confusion_matrix(
            label_mapping, category="training", show=False, title="Confusion Matrix Training")
        CNN.plot_confusion_matrix(
            label_mapping, category="validation", show=False, title="Confusion Matrix Validation")

    elif args.decode:
        CNN.eval()
        CNN2, optimizer, start_epoch, validation_loss = CNN.load_checkpoint(
            optimizer=optimizer, cp=args.load, folder="best")
        CNN2.load_external_test_data(full_test_tensor)
        print(
            f"\nTest accuracy [best training epoch] for participant {args.participant}'s {args.category} data: {CNN2.get_model_accuracy(CNN2)}")
        if args.save:
            CNN.plot_confusion_matrix(
                label_mapping, category="testing", show=False, title="Confusion Matrix Best Epoch")

        CNN3, optimizer, start_epoch, validation_loss = CNN.load_checkpoint(
            optimizer=optimizer, cp=args.load, folder="latest")
        CNN3.load_external_test_data(full_test_tensor)
        print(
            f"\nTest accuracy [last training epoch] for participant {args.participant}'s {args.category} data: {CNN3.get_model_accuracy(CNN3)}")
        if args.save:
            CNN.plot_confusion_matrix(
                label_mapping, category="testing", show=False, title="Confusion Matrix Latest Epoch")
