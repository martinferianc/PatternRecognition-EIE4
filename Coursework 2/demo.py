import numpy as np
from sklearn.datasets import load_iris

# visualisation imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# function to plot the results
def plot(X, Y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    plt.figure(2, figsize=(8, 6))

    # clean the figure
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

#lda.fit(X.T,Y,True)
#X= lda.transform(X.T)

#nca = NCA(max_iter=1000)

#X_train = nca.fit_transform(X_train, Y_train)
#X = nca.transform(X)

#lda = LDA()
#lda.fit(X_train.T,Y_train,True)
#X= lda.transform(X.T)
#print(X)
#print(X)
#plot(X, Y)
