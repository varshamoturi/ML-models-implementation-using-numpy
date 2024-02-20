import numpy as np
from scipy import stats
from collections import Counter
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    if len(x) == 0:
        return 0
    counts = Counter(x)
    return 1 - sum(i ** 2 for i in [i / len(x) for i in counts.values()])



def find_best_split(X, y, loss, min_samples_leaf):
    best_score = float('inf')
    best_split = None

    n_samples, n_features = X.shape

    for col in range(n_features):
        val = np.unique(X[:, col])
        val.sort()

        i = 0
        while i < len(val) - 1:
            split_value = (val[i] + val[i + 1]) / 2

            mask_left = X[:, col] <= split_value
            mask_right = ~mask_left

            if np.any(mask_left) and np.any(mask_right):
                y_left = y[mask_left]
                y_right = y[mask_right]

                score_left = loss(y_left)
                score_right = loss(y_right)
                total_samples = len(y_left) + len(y_right)
                weighted_score = (len(y_left) / total_samples) * score_left + (len(y_right) / total_samples) * score_right

                if weighted_score < best_score:
                    best_score = weighted_score
                    best_split = (col, split_value)

            i += 1

    return best_split


    
class DecisionTree:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        
    def fit(self, X, y):
        """
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.
        """
        if len(set(y)) == 1 or len(y) <= self.min_samples_leaf:
            return self.create_leaf(y)

        l1 = find_best_split(X, y, self.loss, self.min_samples_leaf)

        mask_left = X[:, l1[0]] <= l1[1]
        X_left, y_left = X[mask_left], y[mask_left]

        mask_right = ~mask_left
        X_right, y_right = X[mask_right], y[mask_right]

        return DecisionNode(
            col=l1[0],
            split=l1[1],
            lchild=self.fit_(X_left, y_left),
            rchild=self.fit_(X_right, y_right)
        )

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        
        def _make_single_prediction(val, current_node):
            if isinstance(current_node, LeafNode):
                return current_node.prediction
            
            if val[current_node.col] <= current_node.split:
                return _make_single_prediction(val, current_node.lchild)
            
            return _make_single_prediction(val, current_node.rchild)

        return np.array([_make_single_prediction(i, self.root) for i in X_test])   



class RegressionTree(DecisionTree):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree(DecisionTree):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y, stats.mode(y)[0])
