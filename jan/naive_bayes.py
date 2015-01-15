from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

print(news.data[0])
print(news.target[0], news.target_names)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target,
                                                    test_size=0.25, random_state=0)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

clf_1 = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])
clf_2 = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])

from sklearn.cross_validation import cross_val_score, KFold

def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(scores.mean(), scores.std()))

clfs = [clf_1, clf_2]
for clf in clfs:
    evaluate_cross_validation(clf, news.data, news.target, 5)