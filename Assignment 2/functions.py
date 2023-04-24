import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio

from tqdm import tqdm
from utils import cross_entropy


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.maximum(0, x)


def LoadBatch(filename):
    """ Copied from the dataset website """
    with open('Dataset/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def EvaluateClassifier(X, W1, b1, W2, b2) -> np.array:
    s1 = W1.dot(X.T) + b1
    h = relu(s1)
    s2 = W2.dot(h) + b2
    z = softmax(s2)
    return z


def ComputeCost(X, Y, W1, b1, W2, b2, lamda) -> np.array:
    y_pred = EvaluateClassifier(X, W1, b1, W2, b2)
    ce_loss = cross_entropy(y_pred, Y)
    weight_decay = lamda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    return ce_loss + weight_decay


def ComputeGradsNum(X, Y, W1, b1, W2, b2, lamda, h):
    """ Converted from matlab code """

    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    c = ComputeCost(X, Y, W1, b1, W2, b2, lamda)

    for i in range(len(b1)):
        b_try = np.array(b1)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W1, b_try, W2, b2, lamda)
        grad_b1[i] = (c2-c) / h

    for i in range(len(b2)):
        b_try = np.array(b2)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W1, b1, W2, b_try, lamda)
        grad_b2[i] = (c2-c) / h

    # for i in tqdm(range(W1.shape[0])):
    #     for j in range(W1.shape[1]):
    #         W_try = np.array(W1)
    #         W_try[i, j] += h
    #         c2 = ComputeCost(X, Y, W_try, b1, W2, b2, lamda)
    #         grad_W1[i, j] = (c2-c) / h

    for i in tqdm(range(W2.shape[0])):
        for j in range(W2.shape[1]):
            W_try = np.array(W2)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W1, b1, W_try, b2, lamda)
            grad_W2[i, j] = (c2-c) / h

    return [grad_W1, grad_b1, grad_W2, grad_b2]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in tqdm(range(W.shape[0])):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i,j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i,j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]


def montage(W):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i*5+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()


def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    sio.savemat(f'{name}.mat')
