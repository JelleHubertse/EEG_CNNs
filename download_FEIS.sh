#!/bin/sh

if [ -d "./FEIS" ] 
then
    echo "FEIS directory already exists. To re-download, remove or rename current FEIS directory" 
else
    git clone https://github.com/scottwellington/FEIS
    mv ./FEIS/code_classification/make_svm_features.py ./FEIS/code_classification/make_svm_features_original.py 
    mv ./make_svm_features_altered.py ./FEIS/code_classification/make_svm_features.py
fi