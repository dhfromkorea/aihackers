import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print(digits.data[0], digits.target[0])

def plot_pca_scatter():
	colors = ['black', 'blue', 'purple', 'yellow', 'white', 
          'red', 'lime', 'cyan', 'orange', 'gray']
	for i in range(len(colors)):
			px = X_pca[:, 0][y_digits == i]
			py = X_pca[:, 1][y_digits == i]
			plt.scatter(px, py, c=colors[i])
	plt.legend(digits.target_names)
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()

from sklearn.decomposition import PCA
estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(X_digits)
plot_pca_scatter()