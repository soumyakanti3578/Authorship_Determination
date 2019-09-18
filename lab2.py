__author__ = "Soumyakanti Das"

"""
CSCI 630 lab 2
This program classifies words of length 250 into their authors - Arthur Conan
Doyle, Herman Melville, or Jane Austen. This file mainly contains code to
preprocess data.
"""

import os
import string
from collections import Counter
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
import unicodedata
import sys
import classifiers
import argparse
import pickle

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train models on dataset",
    action="store_true")
parser.add_argument("-d", "--dtree", help="flag to train only decision tree",
    action="store_true")
parser.add_argument("--print", help="prints preorder traversal of decision tree",
    action="store_true")
parser.add_argument("--max_depth", type=int, help="max depth of Decision Tree",
    default=5)
parser.add_argument("-l", "--logreg", help="flag to train only logistic regression",
    action="store_true")
parser.add_argument("--predict", type=str, help="predicts author on provided file",
    metavar=("FILENAME"))
parser.add_argument("--train_test", help="train models on dataset",
    action="store_true")

args = parser.parse_args()

# Assert that the code is run in one of the three modes
assert args.train or args.predict or args.train_test, \
"Provide --train or --predict"

# punctuation table for unicode data
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))

def create_dataset(path):
    """
    Takes a folder path, reads data and creates a dictionary for authors and
    their data.

    :param path: folder path.
    :return author_data: dict of author and their data.
    """
    author_data = dict()
    for (dirpath, dirnames, filenames) in os.walk(path):
#         print(dirpath, dirnames, filenames)
        if filenames:
            data = []
            for file in filenames:
                data.extend(read_file(dirpath + "\\" + file))
            author_data[dirpath.split("\\")[-1]] = data

    return author_data

def read_file(path):
    """
    Reads a text file and returns a list of words in the file.

    :param path: path to the file to read.
    :return data: list of words in the file.
    """
    # table = str.maketrans({key: None for key in string.punctuation})
    data = []
    with open(path, encoding="utf8") as f:
        for line in f:
            for word in line.strip("\n").split():
                if word:
                    data.append(word)

    return data

def stack_dataset(dataset, length=250):
    """
    Takes a dictionary of author and their data as input and stacks the data
    into a 2D array of given length in the 1st axis.

    :param dataset: dictionary of author and their list of words.
    :param length: length of each row in the stacked dataset.
    :return result: dictionary of authors and their 2D array.
    """
    result = dict()
    for author in dataset:
        data = np.array(dataset[author])
        rows = len(data)//length
#         np.random.shuffle(data)
        data = data[:rows*length]
        data = data.reshape((rows, length))
        np.random.shuffle(data)
        result[author] = data

    return result

def remove_punctuation(row):
    """
    Removes punctuation from a list of words.

    :param row: list of words.
    :return result: list of words with no punctuations.
    """
    result = [word.translate(tbl) for word in row]
    return np.array(result)

def count_punctuation(row):
    """
    Counts number of punctuations in the row. Only counts ; and :

    :param row: list of words.
    :return int: count of punctuations.
    """
#     punctuations = ",.;:!?"
# ;: increases accuracy by 10%
# ; - 60%
    punctuations = ";:"
    count_punc= 0
    for word in row:
        for c in word:
            if c in punctuations:
                count_punc += 1
    return count_punc

def average_word_length(counter):
    """
    Takes a Counter dict of words and returns average word length.

    :param counter: Counter dict of words in a row.
    :return double: average word length in the row.
    """
#     len_array = np.array([len(word) for word in remove_punctuation(row)])
    len_array = np.array([len(word)*counter[word] for word in counter])

    return np.mean(len_array)

def get_unique_words(counter):
    """
    Returns the number of unique words in a row.

    :param counter: Counter dict of words in a row.
    :return int: number of unique words.
    """
    return len(counter)

def word_freq(counter, word):
    """
    Returns frequency of a word in a row.

    :param counter: Counter dict of a row.
    :param word: a particular word.
    :return int: count of word in a row.
    """
#     counter = Counter(remove_punctuation(row))
    return counter[word]

def count_words_above(counter, n):
    count = 0
    for word in counter:
        if len(word) > n:
            count += 1

    return count


def word_vector_cosine(counter):
    """
    Returns cosine distance of the row counter vector with a predefined counter
    vector.

    :param counter: Counter dict of the row.
    :return double: cosine distance.
    """
    # Selected words based on their popularity. These words give the best result
    words = ["the", "and", "which", "but", "their", "mrs", "that", "not", "to", "she"]
    # Calculate count of the words in the row and create a "vector" of counts.
    vector = np.array([counter[word] for word in words])
    # Normalize the vector.
    vector_sum = np.sum(vector)
    vector = np.divide(vector, vector_sum)

#     standard = np.divide(np.ones(len(words)), len(words))
# and
    # This is the standard vector which has 1 for the indices of ["she", "their", "mrs"]
    # and 0 everywhere else.
    standard = np.array([1 if words[i] in ["she", "their", "mrs"] else 0 for i in range(len(words))])
    standard = np.divide(standard, np.sum(standard))
    return cosine(vector, standard)

def features_with_punctuations(row):
    """
    calculates features that depend on punctuations.

    :param row: list of words with punctuations.
    :return int: count of punctuations.
    """
    return count_punctuation(row)

def features_without_punctuations(row):
    """
    Calculates features without punctuations.

    :param row: list of words.
    :return tuple: tuple of features.
    """
    row = np.array([word.lower() for word in row])
    counter = Counter(remove_punctuation(row))
    result = average_word_length(counter), get_unique_words(counter), \
                word_freq(counter, "she"), word_freq(counter, "the"), \
                word_vector_cosine(counter), count_words_above(counter, 8)

    return result

def processed_dataset(stacked_dataset):
    """
    Creates a feature set using the stacked dataset.

    :param stacked_dataset: dict of authors and their rows.
    :return features: 2D array of features and target columns.
    """
    author_index = {"acd": 0, "hm": 1, "ja": 2}
    features = []
    for author in stacked_dataset:
        for row in stacked_dataset[author]:
            features.append([features_with_punctuations(row), *features_without_punctuations(row), \
                             author_index[author]])
    features = np.array(features)
    np.random.shuffle(features)

    return features

def min_max_scaler(features, train=True, params=None):
    """
    Scales the features using min and max values.

    :param features: 2D array of features and target.
    :param train: True when in training mode.
    :param params: Used for scaling a single row. Contains (min, max) of
        training data.
    :return None: Scales inplace
    """
    if train:
        params = []
        for col in range(features.shape[1]-1):
            max_val = np.max(features[:, col])
            min_val = np.min(features[:, col])
            params.append((max_val, min_val))
            features[:, col] = (features[:, col] - min_val) / (max_val - min_val)

        return params
    else:
        for col in range(len(features)):
            max_val = params[col][0]
            min_val = params[col][1]
            features[col] = (features[col] - min_val) / (max_val - min_val)


def train(clfs):
    """
    trains classifiers with the data in folder "data".

    :param clfs: list of classifiers
    :return None:
    """
    dataset = create_dataset("data")
    stacked_dataset = stack_dataset(dataset)
    for author in stacked_dataset:
        stacked_dataset[author] = stacked_dataset[author][:2000, :]
    features = processed_dataset(stacked_dataset)
    scale_params = min_max_scaler(features)
    for clf in clfs:
        print("Training " + clf.name)
        clf.scale_params = scale_params
        clf.fit(features[:, :-1], features[:, -1])


def test(filepath, clf):
    """
    Tests a file on a classifier.

    :param filepath: path to the test file.
    :param clf: a classifier.
    :return int: author index.
    """
    row = read_file(filepath)
    assert len(row) >= 250, "Please provide a file with 250 words."
    row = row[:250]
    row = np.array([features_with_punctuations(row), \
     *features_without_punctuations(row)])
    min_max_scaler(row, train=False, params=clf.scale_params)

    return clf.predict(row)

def main():
    """
    Acts as a main method.
    """
    columns = ["count_punc", "mean_word_len", "vocab", "word_freq_she", \
    "word_freq_the", "cosine", "large_words"]

    clfs = []
    if args.train:
        # Create classifiers.
        dtree_clf = classifiers.Decision_Tree(max_depth=args.max_depth)
        lr_clf = classifiers.LogisticMulti()

        if args.dtree:
            clfs.append(dtree_clf)
        if args.logreg:
            clfs.append(lr_clf)
        if not (args.dtree or args.logreg):
            clfs = [dtree_clf, lr_clf]
        # train classifiers
        train(clfs)

        # pickle classifiers
        if args.dtree:
            with open("dtree.obj", "wb") as f:
                pickle.dump(dtree_clf, f)
            if args.print:
                classifiers.print_preorder(dtree_clf.root, columns)
        if args.logreg:
            with open("lr.obj", "wb") as f:
                pickle.dump(lr_clf, f)
        if not (args.dtree or args.logreg):
            with open("dtree.obj", "wb") as f:
                pickle.dump(dtree_clf, f)
            if args.print:
                classifiers.print_preorder(dtree_clf.root, columns)
            with open("lr.obj", "wb") as f:
                pickle.dump(lr_clf, f)

    if args.predict:
        test_file = args.predict
        # clfs will be empty when --train is not provided.
        if not clfs:
            # unpickle classifiers
            if args.dtree:
                with open("dtree.obj", "rb") as f:
                    dtree_clf = pickle.load(f)
                    clfs.append(dtree_clf)
            if args.logreg:
                with open("lr.obj", "rb") as f:
                    lr_clf = pickle.load(f)
                    clfs.append(lr_clf)
            if not (args.dtree or args.logreg):
                with open("dtree.obj", "rb") as f:
                    dtree_clf = pickle.load(f)
                    clfs.append(dtree_clf)
                with open("lr.obj", "rb") as f:
                    lr_clf = pickle.load(f)
                    clfs.append(lr_clf)
        # predict using the unpickled classifiers
        for clf in clfs:
            prediction = test(test_file, clf)
            authors = \
            {0: "Arthur Conan Doyle",
             1: "Herman Melville",
             2: "Jane Austen"}
            print("{} predicts {}".format(clf.name, authors[prediction]))

    # This code can be used to test accuracy of the classifiers
    if args.train_test:
        dataset = create_dataset("data")
        stacked_dataset = stack_dataset(dataset)
        for author in stacked_dataset:
            stacked_dataset[author] = stacked_dataset[author][:2000, :]
        features = processed_dataset(stacked_dataset)
        min_max_scaler(features)
        X_train, X_test, y_train, y_test = \
         train_test_split(features[:, :-1], features[:, -1], test_size=0.3)

        dtree_clf = classifiers.Decision_Tree(max_depth=args.max_depth)
        dtree_clf.fit(X_train, y_train)
        if args.print:
            classifiers.print_preorder(dtree_clf.root, columns)
        print("DTREE accuracy: {}, class_accuracy: {}".
            format(*dtree_clf.score(X_test, y_test)))

        lr_clf = classifiers.LogisticMulti()
        lr_clf.fit(X_train, y_train)
        print("LR accuracy: {}, class_accuracy: {}".
            format(*lr_clf.score(X_test, y_test)))

if __name__ == '__main__':
    main()
