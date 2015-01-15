from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
dataset = fetch_20newsgroups(subset='all', categories=None,
                             shuffle=True, random_state=42)
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
labels = dataset.target
true_k = np.unique(labels).shape[0]
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
# return term document matrix
X = vectorizer.fit_transform(dataset.data)
print("n_samples: %d, n_features: %d" % X.shape)

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=False)
km.fit(X)
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#print(order_centroids)
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
