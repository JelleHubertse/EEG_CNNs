# EEG_CNNs by B149849

## MSc. Speech & Language Processing @ University of Edinburgh

This repository contains the code written for my disseration titled: 'In pursuit of brain-computer interfaces using imagined speech: investigating CNN-based strategies for classification from EEG of heard, imagined and spoken phones'.

### Dataset

The research makes use of the FEIS dataset (V1.1) [1], originally collected by S. Wellington and J. Clayton, at the University of Edinburgh. The FEIS dataset can be obtained using the `download_FEIS.sh` script in this repository. The papers that Wellington [2] and Clayton [3] have written about the dataset, as well as metadata and materials for data processing can be found in that same repository.

### Files

The files contained in this repository are structured as follows:

- `EEG_data.py` is a class that simplifies loading the data provided in the FEIS dataset.
- `ICA.py` is a class that can be used to apply the independent component analysis.
- `utils.py` contains helper code for construction of confusion matrices
- `make_svm_features_altered.py` is a changed version of one of the original FEIS scripts. Running `download_FEIS.sh` will replace the original one with the altered one. The original will remain available.
- `baseline_SVM.py` contains the code written for the baseline SVMs
- Code for the CNNs are split in six different files; three for the 2DCNN and three for the 1DCNNs. By default, the 1DCNN files are configured for the shallow 1DCNN version described in the dissertation. Note that the 1DCNN files need manual alterations to construct the other two 1DCNN model architectures. The functions of interest are the `init` and `forward` functions.
  - *_single.py files are for subject-dependent experiments.
  - *_resample.py files are for the subject-dependent experiments with resampled (balanced) datasets.
  - *_all_data.py files are for the subject-independent experiments.
- `run_experiments.sh` contains a nested-loop infrastructure that allows (bulk) running of experiments.

You will find lines of code pointing to expansion of model options for voicing and phoneme classification. These are currently not stable (yet), and are to be attempted at your own risk.

The code assumes `python 3.7.4` and `conda 4.8.3`. Dependencies include:

- mne==0.20.7
- scikit-learn==0.23.1
- torch==1.5.1
- pandas==1.0.5
- numpy==1.19.1
- matplotlib==3.2.2
- entropy --> To use entropy/time series features, follow the installation instructions for this package on GitHub: https://github.com/raphaelvallat/entropy

### References

1. S. Wellington, J. Clayton, "Fourteen-channel EEG with Imagined Speech (FEIS) dataset," v1.0, University of Edinburgh, Edinburgh, UK, 2019. doi:10.5281/zenodo.3369178
2. Wellington, S. (2019). An investigation into the possibilities and limitations of decoding heard, imagined and spoken phonemes using a low-density, mobile EEG headset [Master Thesis]. University of Edinburgh.
3. Clayton, J. (2019). Towards Phone Classification from Imagined Speech Using a Lightweight EEG Brain-Computer Interface [Master Thesis]. University of Edinburgh.
