import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

boston = load_boston()

print(boston.feature_names)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, 
													test_size=0.25, random_state=33)
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import preprocessing

clf = Pipeline([
	('scaler', preprocessing.StandardScaler()),
	('linear_model', linear_model.SGDRegressor(loss='squared_loss', penalty=None,  random_state=42))
	])
cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
scores = cross_val_score(clf, X_train, y_train, cv=cv)
print(scores, scores.mean(), scores.std())

from sklearn import svm
clf_svr = Pipeline([
	('scaler', preprocessing.StandardScaler()),
	('svm', svm.SVR(kernel='linear'))
	])
scores = cross_val_score(clf_svr, X_train, y_train, cv=cv)
print(scores, scores.mean(), scores.std())

clf.fit(X_train, y_train)
clf.predict(X_test) == y_test
