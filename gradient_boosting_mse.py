import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_dataset(path):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    y = dataset[:, -1]
    X = dataset[:, 0:- 1]
    return X, y

def gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1):
    """Given X, a array y and num_iter return y_mean and trees 
   
    Input: X, y, num_iter
           max_depth
           nu (is the shinkage)
    Outputs:y_mean, array of trees from DecisionTreeRegression
    """
    trees = []
    N, _ = X.shape
    y_mean = np.mean(y)
    fm = y_mean
    for _ in range(num_iter):
        res = y-fm
        tree = DecisionTreeRegressor(max_depth=max_depth)
        # Fit the decision tree on the weighted data
        tree.fit(X, res)
        y_pred = tree.predict(X)
        fm+= nu*y_pred
        trees.append(tree)
    return y_mean, trees  

def gradient_boosting_predict(X, trees, y_mean,  nu=0.1):
    """Given X, trees, y_mean predict y_hat
    """
    y_hat = np.full(X.shape[0], y_mean)
    for tree in trees:
        y_hat += nu * tree.predict(X)
    return y_hat

