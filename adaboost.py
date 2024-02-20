import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    dataset = np.loadtxt(filename, delimiter=",")
    Y = dataset[:, -1]
    X = dataset[:, 0:- 1]
    Y[Y==0] = -1
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N) / N
    for _ in range(num_iter):
        # Create a decision tree classifier with given max_depth
        tree = DecisionTreeClassifier(max_depth=max_depth)
        # Fit the decision tree on the weighted data
        tree.fit(X, y, sample_weight=d)
        # Predictions based on the current tree
        y_pred = tree.predict(X)
        # Calculate weighted error
        err = np.sum(d[y != y_pred]) / np.sum(d)
        # Compute tree weight
        alpha = np.log((1 - err) / err)
        # Update sample weights
        d *= np.exp(alpha * y * y_pred)
        # Append the tree and its weight to the lists
        trees.append(tree)
        trees_weights.append(alpha)
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    for tree, alpha in zip(trees, trees_weights):
        # Predict using each tree
        y += alpha * tree.predict(X)
    # Convert predictions to binary labels {-1, 1}
    y = np.sign(y)
    return y
