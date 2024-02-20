# Machine Learning Toolbox: Implementations of Regression and Classification Algorithms

This repository contains Python implementations of various machine learning algorithms for both regression and classification tasks.

## Files

- **linreg.py**: Contains the implementations of Linear Regression, Logistic Regression, and Ridge Regression classes.
- **bayes.py**: Contains the implementation of the Naive Bayes classifier and helper functions for data preprocessing.
- **dtree.py**: Contains the implementations of Decision Tree classes for regression and classification tasks.
- **rf.py**: Contains the implementations of Random Forest classes for regression and classification tasks.
- **gradient_boosting_mse.py**: Contains the implementations of the Gradient Boosting algorithm for regression tasks.
- **README.md**: This file.

## Linear Regression (LinearRegressionNP)

The `LinearRegressionNP` class implements linear regression using gradient descent optimization. It includes methods for fitting the model to training data and making predictions on new data.

### Usage

```python
from linear_logistic_regression_np import LinearRegressionNP

# Instantiate the Linear Regression model
model = LinearRegressionNP(eta=0.00001, lmbda=0.0, max_iter=1000)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)
```

## Logistic Regression (LogisticRegressionNP)

The `LogisticRegressionNP` class implements logistic regression using gradient descent optimization. It includes methods for fitting the model to training data, making predictions on new data, and computing probabilities.

### Usage

```python
from linear_logistic_regression_np import LogisticRegressionNP

# Instantiate the Logistic Regression model
model = LogisticRegressionNP(eta=0.00001, lmbda=0.0, max_iter=1000)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Get probabilities
probabilities = model.predict_proba(X_test)
```

## Ridge Regression (RidgeRegressionNP)

The `RidgeRegressionNP` class implements ridge regression using gradient descent optimization. It includes methods for fitting the model to training data and making predictions on new data.

### Usage

```python
from linear_logistic_regression_np import RidgeRegressionNP

# Instantiate the Ridge Regression model
model = RidgeRegressionNP(eta=0.00001, lmbda=0.0, max_iter=1000)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)
```
## NaiveBayes (NaiveBayesNP)
### Usage

### Training the Model

```python
from naive_bayes_np import NaiveBayesNP

# Instantiate the Naive Bayes model
model = NaiveBayesNP()

# Fit the model to training data
model.fit(X_train, y_train)
```

### Making Predictions

```python
# Make predictions on new data
predictions = model.predict(X_test)
```

### Cross-Validation

```python
from naive_bayes_np import kfold_CV

# Perform k-fold cross-validation
accuracies = kfold_CV(model, X, y, k=4)
```
## Decision Tree Implementations

### DecisionNode Class

Represents a decision node in the decision tree.

### LeafNode Class

Represents a leaf node in the decision tree.

### `gini()` Function

Calculates the Gini impurity score for a set of values.

### `find_best_split()` Function

Finds the best split for a given dataset based on the loss function and minimum samples per leaf.

### `DecisionTreeNP` Class

Base class for decision trees. Provides methods for fitting the model to data and making predictions.

### `RegressionTreeNP` Class

Subclass of `DecisionTreeNP` for regression tasks. Uses variance as the loss function.

### `ClassifierTreeNP` Class

Subclass of `DecisionTreeNP` for classification tasks. Uses Gini impurity as the loss function.

### Usage

### Regression Tree

```python
from decision_tree_np import RegressionTreeNP

# Instantiate the Regression Tree model
model = RegressionTreeNP(min_samples_leaf=1)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Evaluate the model
r2_score = model.score(X_test, y_test)
```

### Classification Tree

```python
from decision_tree_np import ClassifierTreeNP

# Instantiate the Classification Tree model
model = ClassifierTreeNP(min_samples_leaf=1)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
```

## Random Forest Implementations

### `RandomForestNP` Class

Base class for random forests. Provides methods for fitting the model to data and making predictions.

### `RandomForestRegressorNP` Class

Subclass of `RandomForestNP` for regression tasks. Uses a collection of `RegressionTreeNP` instances to construct the random forest.

### `RandomForestClassifierNP` Class

Subclass of `RandomForestNP` for classification tasks. Uses a collection of `ClassifierTreeNP` instances to construct the random forest.

### Usage

### Random Forest for Regression

```python
from random_forest_np import RandomForestRegressorNP

# Instantiate the Random Forest model
model = RandomForestRegressorNP(n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Evaluate the model
r2_score = model.score(X_test, y_test)
```

### Random Forest for Classification

```python
from random_forest_np import RandomForestClassifierNP

# Instantiate the Random Forest model
model = RandomForestClassifierNP(n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
```
## Adaboost Implementation

### `adaboost` Function

The `adaboost` function takes input data `X` and labels `y`, the number of iterations `num_iter`, and the maximum depth of the decision trees as parameters. It returns a list of decision trees and a list of corresponding weights.

### `adaboost_predict` Function

The `adaboost_predict` function takes input data `X`, a list of decision trees, and a list of weights as parameters. It returns the predicted labels for the input data.

### Utility Functions

### `accuracy` Function

The `accuracy` function calculates the accuracy of predicted labels compared to true labels.

### `parse_spambase_data` Function

The `parse_spambase_data` function reads data from a file and returns feature vectors `X` and labels `y`.

###  Usage

```python
import numpy as np
from adaboost import adaboost, adaboost_predict, accuracy, parse_spambase_data

# Load data
X, y = parse_spambase_data("spambase.data")

# Split data into training and testing sets
num_train = int(len(X) * 0.8)
X_train, X_test = X[:num_train], X[num_train:]
y_train, y_test = y[:num_train], y[num_train:]

# Train Adaboost model
num_iter = 50
trees, weights = adaboost(X_train, y_train, num_iter=num_iter)

# Make predictions
y_pred_train = adaboost_predict(X_train, trees, weights)
y_pred_test = adaboost_predict(X_test, trees, weights)

# Calculate accuracy
train_acc = accuracy(y_train, y_pred_train)
test_acc = accuracy(y_test, y_pred_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```
## Gradient Boosting Implementation

### `gradient_boosting_mse` Function

The `gradient_boosting_mse` function takes input data `X` and labels `y`, the number of iterations `num_iter`, the maximum depth of the decision trees, and the learning rate `nu` as parameters. It returns the mean of the labels `y_mean` and a list of decision trees.

### `gradient_boosting_predict` Function

The `gradient_boosting_predict` function takes input data `X`, a list of decision trees, the mean of the labels `y_mean`, and the learning rate `nu` as parameters. It returns the predicted labels `y_hat` for the input data.

### Utility Functions

### `load_dataset` Function

The `load_dataset` function reads data from a file and returns feature vectors `X` and labels `y`.

###  Usage

```python
import numpy as np
from gradient_boosting import gradient_boosting_mse, gradient_boosting_predict, load_dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_dataset("dataset.csv")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
num_iter = 100
max_depth = 3
nu = 0.1
y_mean, trees = gradient_boosting_mse(X_train, y_train, num_iter=num_iter, max_depth=max_depth, nu=nu)

# Make predictions
y_pred_train = gradient_boosting_predict(X_train, trees, y_mean, nu=nu)
y_pred_test = gradient_boosting_predict(X_test, trees, y_mean, nu=nu)

# Calculate R^2 score
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Train R^2 Score: {train_r2:.4f}")
print(f"Test R^2 Score: {test_r2:.4f}")
```

## Requirements

- NumPy
- pandas (only required for data preprocessing, not for the main implementations)
- scikit-learn
- SciPy
  
## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.
