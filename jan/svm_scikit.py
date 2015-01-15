"""
One of the greatest advantages of SVM is that they are very effective 
when working on high-dimensional spaces,that is, on problems which have a lot 
of features to learn from. They are also very effective when the 
data is sparse (think about a high-dimensional space with very few instances). 
Besides, they are very efficient in terms of memory storage, since only a 
subset of the points in the learning space is used to represent the decision surfaces.
"""

import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# face dataset
faces = fetch_olivetti_faces()


def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1,
                        bottom=0, top=1,
                        hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
    plt.show()

print_faces(faces.images, faces.target, 20)

from sklearn.svm import SVC

svc_1 = SVC(kernel='linear')
print(svc_1)
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target,
                                                    test_size=0.25, random_state=0)

cv = KFold(len(y_train), 5, shuffle=True, random_state=0)
scores = cross_val_score(svc_1, X_train, y_train, cv=cv)
print(scores.mean(), scores.std())