__author__ = "Soumyakanti Das"

"""
CSCI 630 lab 2

This file contains code for Decision Tree and Multi Logistic Regression
classifiers.
"""

import numpy as np

class Question:
    """
    Question is used to store the various questions a decision node asks.
    """
    def __init__(self, col, value):
        """
        init method.

        :param col: column number.
        :param value: split value.
        """
        self.col = col
        self.value = value

    def pretty_print(self, columns):
        """
        Used to pretty print Decision Tree.

        :param columns: features of training set.
        """
        return columns[self.col] + " <= " + str(self.value) + " ?"

class Leaf:
    """
    Represents the Leaf nodes of a Decision Tree.
    """
    def __init__(self, arr):
        """
        :param arr: features, 2D array.
        """
        # class counts.
        self.counts = self.count(arr[:, -1])
        # leaf prediction based on max class value.
        self.pred = self.prediction()

    def count(self, row):
        """
        Returns class counts.

        :param row: target column.
        :return list: class counts.
        """
        counts = [0, 0, 0]
        for i in row:
            counts[int(i)] += 1

        return counts

    def prediction(self):
        """
        Returns prediction of the Leaf.
        """
        # get max value
        max_count = max(self.counts)
        # list of all candidates with maximum value
        candidates = []
        # fill the candidate list
        for i in range(len(self.counts)):
            if self.counts[i] == max_count:
                candidates.append(i)

        if len(candidates) == 1:
            return candidates[0]
        # Return random candidate if multiple candidates.
        return candidates[np.random.randint(len(candidates), size=1)[0]]

class Node:
    """
    Represents inner nodes of Decision Tree.
    """
    def __init__(self, question, true, false):
        """
        :param question: Question object for the node.
        :param true: features for which question is true.
        :param false: features for which question is false.
        """
        self.question = question
        self.true = true
        self.false = false

class Decision_Tree:
    """
    Decision Tree classifier.
    """
    def __init__(self, max_depth=5):
        """
        :param max_depth: maximum depth of the tree.
        """
        self.max_depth = max_depth
        self.root = None
        self.name = "Decision Tree Classifier"
        # list of (min max) for each feature. Stored here while training, to be
        # used for testing row.
        self.scale_params = None

    def fit(self, X, y):
        """
        Fits the model.
        :param X: features.
        :param y: target column.
        :return None
        """
        y = y.reshape(y.shape[0], 1)
        self.root = self.build_tree(np.append(X, y, 1), self.max_depth)

    def build_tree(self, array, depth=5):
        """
        Builds a tree from the feature set.
        """
        # base condition wrt the depth of the tree.
        if depth == 0:
            return Leaf(array)

        # Find the best split of the array, i.e., split for which gain is max.
        # t, f are true and false split for the question.
        gain, question, t, f = best_split(array)

        # Here gain threshold could be applied, but I didn't add a threshold
        # for entropy or gain.
        if gain == 0:
            return Leaf(array)

        # recursively build tree for true and false split.
        true = self.build_tree(t, depth-1)
        false = self.build_tree(f, depth-1)

        # Return a Node object with the question, true and false objects.
        return Node(question, true, false)

    def predict(self, row):
        """
        Predicts authorship for a row.

        :param row: testing row.
        :return int: author index.
        """
        # Traverse the tree asking questions and checking the row for each
        # question.
        node = self.root
        while isinstance(node, Node):
            if row[node.question.col] <= node.question.value:
                node = node.true
            else:
                node = node.false

        # Here node is a Leaf object.
        return node.pred

    def score(self, x, y):
        """
        Scores multiple testing rows.

        :param x: features with multiple rows.
        :param y: corresponding targets.
        :return tuple: accuracy, class accuracy.
        """
        y = y.reshape(y.shape[0], 1)
        correct = 0
        correct_counts = [0]*3
        _, counts = np.unique(y, return_counts=True)
        for i in range(x.shape[0]):
            if self.predict(x[i, :]) == y[i, 0]:
                correct += 1
                correct_counts[int(y[i, 0])] += 1
        return correct / x.shape[0], list(np.divide(np.array(correct_counts), counts))

def print_preorder(root, columns, spacing=""):
    """
    prints the decision tree with preorder traversal.

    :param root: root node of the tree.
    :param columns: list of feature names.
    :param spacing: used for pretty printing
    :return None
    """
    if isinstance(root, Leaf):
        print(spacing + str(root.counts) + "\tpred: " + str(root.pred))
    else:
        print(spacing + root.question.pretty_print(columns))
        print_preorder(root.true, columns, spacing=spacing+"\tT: ")
        print_preorder(root.false, columns, spacing=spacing+"\tF: ")


def entropy(arr):
    """
    Calculates entropy on a split.

    :param arr: 2D array of features.
    :return double: entropy
    """
    uniq_values, counts = np.unique(arr[:, -1], return_counts=True)
    total_count = np.sum(counts)
    probs = counts / total_count
    plogp = [-p*np.log2(p) for p in probs]
    return sum(plogp)

def get_splits(feature):
    """
    Returns all possible splits given a column.

    :param feature: a feature column.
    :return array: array of splits.
    """
    # Get unique values from the column in sorted order
    uniq = np.unique(feature)
    result = []
    # For each pair, calculate mean and append it to result.
    for i in range(0, len(uniq)-1):
        result.append((uniq[i] + uniq[i+1])/2)

    return np.array(result)

def best_split(arr):
    """
    Finds best split and calculates gain for the split.

    :param arr: 2D array of features.
    :return tuple: gain, question, true and false splits.
    """
    best_gain = 0
    best_ques = None
    true_split, false_split = None, None
    # calculate the entropy before spliting, E_init.
    entropy_init = entropy(arr)
    # For each column, get split points
    for col in range(arr.shape[1]-1):
        splits = get_splits(arr[:, col])
        for split in splits:
            # Split dataset into true and false
            true_data = arr[arr[:, col] <= split]
            false_data = arr[arr[:, col] > split]

            # continue because for this gain will be zero.
            if len(true_data) == 0 or len(false_data) == 0:
                continue

            # calculate gain using the true and false splits.
            gain = entropy_init - (1/arr.shape[0])* \
                    (true_data.shape[0] * entropy(true_data) + \
                    false_data.shape[0]*entropy(false_data))

            # Calculate maximum gain, and question and splits for that gain.
            if gain >= best_gain:
                best_gain = gain
                best_ques = Question(col, split)
                true_split = true_data
                false_split = false_data

    return best_gain, best_ques, true_split, false_split

class LogisticMulti:
    """
    Logistic regression implementation for multiple classes using one vs all
    strategy.
    """
    def __init__(self, lr=0.001, iters=20000):
        """
        :param lr: learning rate.
        :param iters: number of iterations.
        """
        self.lr = lr
        self.iters = iters
        # Learned weights for each author.
        self.authors = dict()
        self.name = "Logistic Regression Classifier"
        # list of (min max) for each feature. Stored here while training, to be
        # used for testing row.
        self.scale_params = None

    def fit(self, X, y):
        """
        Fits the model.

        :param X: features.
        :param y: target column.
        :return None
        """
        # Add a new column in the beginning with 1s for bias.
        X = np.insert(X, 0, 1, axis=1)

        # For each author, convert target column for one vs all training.
        for auth in np.unique(y):
            # Create a copy of target column with 1s for positive samples and
            # 0s for negative samples.
            y_copy = np.where(y == auth, 1, 0)
            y_copy = y_copy.reshape(y.shape[0], 1)
            # Initialize weights as 1s
            w = np.ones((1, X.shape[1]))

            # Gradient descent with log likelihood
            for _ in range(self.iters):
                error_vector = y_copy - self.sigmoid(X, w)
                w += self.lr * np.matmul(np.transpose(error_vector), X)

            self.authors[auth] = w

    def sigmoid(self, X, w):
        """
        Returns sigmoid of X with given weights.

        :param X: 2D array of features.
        :param w: learned weights
        """
        return 1 / (1 + np.exp(-np.matmul(X, np.transpose(w))))

    def predict(self, row):
        """
        Predicts authorship for a row.

        :param row: testing row.
        :return int: author index.
        """
        row = np.insert(row, 0, 1)
        # predictions for each author
        predictions = dict()
        for key in self.authors:
            out = self.sigmoid(row, self.authors[key])
            predictions[key] = out
        prediction, val = None, -1

        # Finds the author for which the sigmoid activation is maximum.
        for key in predictions:
            if predictions[key] > val:
                val = predictions[key]
                prediction = key

        return prediction

    def score(self, X, y):
        """
        Scores multiple testing rows.

        :param X: features with multiple rows.
        :param y: corresponding targets.
        :return tuple: accuracy, class accuracy.
        """
        correct = 0
        correct_counts = [0]*3
        _, counts = np.unique(y, return_counts=True)
        for i in range(X.shape[0]):
            if self.predict(X[i, :]) == y[i]:
                correct += 1
                correct_counts[int(y[i])] += 1

        return correct / len(y), list(np.divide(np.array(correct_counts), counts))
