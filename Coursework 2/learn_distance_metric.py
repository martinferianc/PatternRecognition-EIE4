import numpy as np
from tqdm import tqdm
import os
import copy
from lda import wPCA

WEIGHT_DIR = "weights/"

def f(X, A):
    # Initialize the sum
    s = 0
    # Initalize the array remembering which pairs were done
    c = []
    # Iterate over all the data
    for i in range(len(X)):
        for j in range(len(X)):
            # Make sure that f is not calculated with respect to itself
            if i!=j and ([i,j] not in c and [j,i] not in c):
                s+= np.sqrt(np.matmul(np.matmul((X[i,:]-X[j,:]).T,A),(X[i,:]-X[j,:])))
                c.append([i,j])
    return s


def grad_f(X, A):
    scaling_factor = (1/2)*(1/f(X,A))
    # Initialize the sum
    s = np.zeros((X.shape[1],X.shape[1]))
    # Initalize the array remembering which pairs were done
    c = []
    for i in range(len(X)):
        for j in range(len(X)):
            # Make sure that f is not calculated with respect to itself
            if i!=j and ([i,j] not in c and [j,i] not in c):
                s+= np.matmul(X[i,:],X[i,:].T) -2*np.matmul(X[i,:],X[j,:].T) +np.matmul(X[j,:],X[j,:].T)
                c.append([i,j])
    return scaling_factor*s




def adam_gradient_descent(func, gradient_func, A, X, iterations = 20):
    alpha =0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    A_0 = A
    A_prev = 0
    m_t = 0
    v_t = 0
    t = 0
    y = func(X, A_0)
    y_prev = 0
    for i in range(1,iterations):
        g_t = gradient_func(X,A_0)
        m_t = beta_1*m_t + (1-beta_1)*g_t
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)
        m_cap = m_t/(1-(beta_1**i))
        v_cap = v_t/(1-(beta_2**i))
        A_prev = copy.deepcopy(A_0)
        A_0 = A_0 - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)
        y = func(X, A_0)
        k = 0
        while np.isnan(y):
            k+=1
            A_0 = A_prev - ((alpha**k)*m_cap)/(np.sqrt(v_cap)+epsilon)
            y = func(X, A_0)
    return A_0

def find_A(X, name, overwrite=False):
    if os.path.exists(WEIGHT_DIR+"{}.npy".format(name)) and overwrite is False:
        return np.load(WEIGHT_DIR+"{}.npy".format(name))

    # Initialize default random PSD A
    _A = np.random.uniform(0,0.01,(X.shape[1],X.shape[1]))
    _A = np.dot(_A,_A.T)
    A = adam_gradient_descent(f, grad_f, _A, X, iterations = 50)

    if overwrite is True:
        np.save(WEIGHT_DIR+"{}.npy".format(name), A)
    return A


def find_U(X, name, overwrite=False):
    if os.path.exists(WEIGHT_DIR+"{}.npy".format(name)) and overwrite is False:
        return np.load(WEIGHT_DIR+"{}.npy".format(name))

    # Initialize default random PSD A
    n = X.shape[0]
    U = wPCA(X.T,n)

    if overwrite is True:
        np.save(WEIGHT_DIR+"{}.npy".format(name), U)
    return U

def find_matrices(X,Y, overwrite=True):
    A_s = {}
    U_s = {}

    classes, counts = np.unique(Y, return_counts = True)
    c = len(classes)
    for i in tqdm(range(c)):
        _X_i = X[np.where(Y == classes[i])]
        U = find_U(_X_i,"U_{}".format(i), overwrite)
        X_i = copy.deepcopy(_X_i)
        X_i = np.matmul(X_i,U)
        A = find_A(X_i,"A_{}".format(i), overwrite)
        A_s[classes[i]] = A
        U_s[classes[i]] = U

    return A_s, U_s


if __name__ == '__main__':
    main()
