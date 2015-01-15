from sklearn import datasets

iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

print('########## Printing the feature names for the dataset')
print(iris.feature_names)
print('######### Printing dataset shape ##########')
print(X_iris.shape, y_iris.shape)
print('######### Printing dataset samples ############')
print(X_iris[0], y_iris[0])

# Learning objective is to predict species given sepal width and length

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

# Extract all rows and first two features.
X, y = X_iris[:, :2], y_iris

# Divide the dataset into train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

import matplotlib.pyplot as plt
scaler = preprocessing.StandardScaler().fit(X_train)

# Plot the features
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
	xs = X_train[:, 0][y_train == i]
	ys = X_train[:, 1][y_train == i]
	plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf = Pipeline([
	('scaler', preprocessing.StandardScaler()),
	('linear_model', SGDClassifier())
	])
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
scores = cross_val_score(clf, X, y, cv=cv)
print('Scores: ', scores)
print('Mean score, std', scores.mean(), scores.std())