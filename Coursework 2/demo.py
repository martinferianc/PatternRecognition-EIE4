import numpy as np
from sklearn.datasets import load_iris

from metric_learn import LFDA
# visualisation imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nca import NCA

from lfda import LFDA
from kernel_lda import LDA

# loading our dataset

iris_data = load_iris()
# this is our data

X = iris_data['data']
# these are our constraints
Y = iris_data['target']

# function to plot the results
def plot(X, Y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.figure(2, figsize=(8, 6))

    # clean the figure
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

lda = LDA()
#lfda = LFDA(k=10)

lda.fit(X_train.T,Y_train,True)
X = lda.transform(X.T)
#X_nca = lfda.fit_transform(X, Y)
print(X.shape, Y.shape)
plot(X, Y)
