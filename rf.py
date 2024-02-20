import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForestNP:
    def __init__(self, n_estimators=10, oob_score=False, model_type='regr'):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data. Keep track of the indexes of
        the OOB records for each tree. After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """

        oob_pred_arr = [[] for _ in range(len(X))]
        oob_score_func = r2_score if isinstance(self, RandomForestRegressor) else accuracy_score

        for i in range(self.n_estimators):
            resampled_indexes = resample(range(len(X)), n_samples=int((2/3)*len(X)), random_state=i, replace=False)
            index_oob = [j for j in range(len(X)) if j not in resampled_indexes]
            tree = self.trees[i]
            tree.fit(X[resampled_indexes], y[resampled_indexes])

            preds = tree.predict(X[index_oob])
            preds = [x.prediction for x in preds]

            for indexes, pred in enumerate(preds):
                oob_pred_arr[index_oob[indexes]].append(pred)

        if self.oob_score:
            oob = [np.mean(sublist) if isinstance(self, RandomForestRegressor) else stats.mode(sublist, keepdims=False)[0] for sublist in oob_pred_arr]
            nan_lst = [i for i in range(len(oob)) if str(oob[i]) == 'nan']
            oob_filtered = [oob[i] for i in range(len(oob)) if i not in nan_lst]
            y_filtered = [y[i] for i in range(len(y)) if i not in nan_lst]
            self.oob_score_ = oob_score_func(y_filtered, oob_filtered)

class RandomForestRegressorNP(RandomForestNP):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = [RegressionTreeNP(min_samples_leaf=min_samples_leaf, max_features=max_features) for _ in range(n_estimators)]

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each tree's prediction by
        the number of observations in the leaf making that prediction. Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = np.zeros(len(X_test))
        total_weights = np.zeros(len(X_test))

        for tree in self.trees:
            leaf_nodes = tree.predict(X_test=X_test)
            leaf_values = np.array([float(leaf.prediction) for leaf in leaf_nodes])
            leaf_weights = np.array([float(leaf.n) for leaf in leaf_nodes])

            predictions += leaf_values * leaf_weights
            total_weights += leaf_weights

        non_zero_denominators = np.where(total_weights != 0, total_weights, 1)
        predictions /= non_zero_denominators

        return predictions
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_test,self.predict(X_test))

class RandomForestClassifierNP(RandomForestNP):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.trees = [ClassifierTreeNP(min_samples_leaf=min_samples_leaf, max_features=max_features) for _ in range(n_estimators)]


    def predict(self, X_test) -> np.ndarray:
        empty = [[]]*len(X_test)
        
        for tree in self.trees:
            nodes = [list(i.y) for i in tree.predict(X_test)]
            empty = [x + y for x, y in zip(empty, nodes)]
        return [stats.mode(sublist, keepdims=True).mode[0] for sublist in empty]
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return accuracy_score(y_test,self.predict(X_test))
