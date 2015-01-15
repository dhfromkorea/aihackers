import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('iris.csv', names=['sl','sw','pt','pw', 'class'], index_col=False)
data_2d = data[['sl','sw','class']]

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_2d[['sl','sw']], data_2d['class'], test_size=0.25, random_state=33)

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf = Pipeline([
	('scaler', preprocessing.StandardScaler()),
	('linear_model', SGDClassifier())
	])
cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
scores = cross_val_score(clf, X_train, y_train, cv=cv)
print('Scores: ', scores)
print('Mean score, std', scores.mean(), scores.std())