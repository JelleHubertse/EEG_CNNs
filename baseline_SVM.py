import numpy as np
import os.path as op
import os
import random
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split


random_seed = 14
np.random.seed(random_seed)
random.seed(random_seed)

experiments_dir = "./experiments"
experiments_list = [entry for entry in os.listdir(experiments_dir)]
npy_dir = "./experiments/svm_features"

clf = svm.SVC(C=1, kernel="linear")

# expeirment and
x = np.load(op.join(npy_dir, f"testing", "speaking.npy"))
y = np.load(op.join(npy_dir, f"testing", "labels.npy"))

x = np.nan_to_num(x)

# print(
#     f"The shape of the data variable x is {x.shape}, representing {x.shape[0]} prompts, {x.shape[1]} features and {x.shape[2]} windows.\n \
#     All {y.shape[0]} labels in y correspond to the {x.shape[0]} prompts in data variable x.\n")

# transforming data variable x into the right form
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

balance_data = False  # set to True for balancing the data in the datasets

vowels = ['goose', 'thought', 'fleece', 'trap']
if balance_data:
    # making the labels binary by letter class and balancing the dataset
    consonants = random.choices(['zh', 'p', 'sh', 'n', 'k',
                                 's', 'ng', 't', 'f', 'z', 'm', 'v'], k=4)+[1]
    #  adding the [1] to not overwrite the vowels

    labels_binary = np.array(
        [1 if label in vowels else label for label in list(y)])
    labels_binary = np.array(
        [0 if label in consonants else label for label in list(labels_binary)])

    mask = [True if label in ["0", "1"] else False for label in labels_binary]
    x = x[mask, :]
    y = y[mask]
    labels_binary = labels_binary[mask]
else:
    consonants = ['zh', 'p', 'sh', 'n', 'k',
                  's', 'ng', 't', 'f', 'z', 'm', 'v']
    labels_binary = [1 if label in vowels else 0 for label in list(y)]


# splitting the test and training sets randomly
x_train, x_test, y_train, y_test = train_test_split(
    x, labels_binary, test_size=0.2, stratify=labels_binary, random_state=random_seed)

# Fit the SVM on the training sets
clf.fit(x_train, y_train)

print(f"Actual labels:\t\t{y_test}")
print(f"predicted labels:\t{clf.predict(x_test)}")
print(f"Score: \t\t\t\t{float(clf.score(x_test, y_test))*100}%")

crossval_test = cross_val_score(clf, x_test, y_test, cv=5)
print(
    f"cross validation test sets:\t\t{cross_val_score(clf, x_test, y_test, cv=5)}")
print("\tAccuracy: %0.2f (+/- %0.2f)" %
      (crossval_test.mean(), crossval_test.std() * 2))
