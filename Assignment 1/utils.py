import pickle
import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_images_and_labels(files) -> dict:
    if isinstance(files, str):
        files = [files]
    data = np.concatenate([unpickle(file)[b'data'] for file in files], axis=0)
    labels = np.concatenate([np.array(unpickle(file)[b'labels']) for file in files], axis=0)
    n_labels = np.unique(labels).size
    one_hot_lables = np.zeros((n_labels, labels.size))
    np.put_along_axis(one_hot_lables, np.expand_dims(labels, axis=0), 1, axis=0)
    return {'data': data, 'labels': labels, 'one_hot_labels': one_hot_lables}


def init_weights(K: int, d: int) -> tuple:
    weights = np.random.normal(0, 1, (K, d))
    bias = np.random.normal(0, 1, (K, 1))
    return weights, bias


def softmax(x):
    """ Standard definition of the softmax function """
    # clip to avoid overflow for float64
    x = np.clip(x, a_min=0, a_max=300)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(y_pred, one_hot_labels):
    # clip to avoid log(0)
    y_pred = np.clip(y_pred, a_min=1e-100, a_max=np.inf)
    n_samples = one_hot_labels.shape[1]
    ce_loss = np.sum(- one_hot_labels * (np.log(y_pred))) / n_samples
    return ce_loss


def sigmoid(x):
    x = np.clip(x, a_min=0, a_max=300)
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(y_pred, one_hot_labels):
    # clip to avoid log(0)
    y_pred = np.clip(y_pred, a_min=1e-100, a_max=np.inf)
    one_minus_y_pred = np.ones_like(y_pred) - y_pred
    one_minus_y_pred = np.clip(one_minus_y_pred, a_min=1e-100, a_max=np.inf)
    n_samples = one_hot_labels.shape[1]
    bce_loss = -np.sum(one_hot_labels * (np.log(y_pred)) +
                (np.ones_like(one_hot_labels) - one_hot_labels) * np.log(one_minus_y_pred)) / n_samples
    return bce_loss


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


