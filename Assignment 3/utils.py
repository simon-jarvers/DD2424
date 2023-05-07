import pickle
import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_images_and_labels(files, first_n_samples=None, last_n_samples=None) -> dict:
    if isinstance(files, str):
        files = [files]
    data = np.concatenate([unpickle(file)[b'data'] for file in files], axis=0)
    labels = np.concatenate([np.array(unpickle(file)[b'labels']) for file in files], axis=0)
    n_labels = np.unique(labels).size
    one_hot_labels = np.zeros((labels.size, n_labels))
    np.put_along_axis(one_hot_labels, np.expand_dims(labels, axis=1), 1, axis=1)
    if first_n_samples is not None and last_n_samples is not None:
        train_split = {'data': data[:first_n_samples, :],
                       'labels': labels[:first_n_samples],
                       'one_hot_labels': one_hot_labels[:first_n_samples, :]}
        val_split = {'data': data[-last_n_samples:, :],
                     'labels': labels[-last_n_samples:],
                     'one_hot_labels': one_hot_labels[-last_n_samples:, :]}
        return train_split, val_split
    if first_n_samples is not None:
        data = data[:first_n_samples, :]
        labels = labels[:first_n_samples]
        one_hot_labels = one_hot_labels[:, :first_n_samples]
    if last_n_samples is not None:
        data = data[last_n_samples:, :]
        labels = labels[last_n_samples:]
        one_hot_labels = one_hot_labels[:, last_n_samples:]
    return {'data': data, 'labels': labels, 'one_hot_labels': one_hot_labels}


def init_weights(input_size: int, output_size: int, sig: float = None) -> tuple:
    sig = np.sqrt(1/input_size) if sig is None else sig
    weights = np.random.normal(0, sig, (input_size, output_size))
    bias = np.random.normal(0, sig, (1, output_size))
    return weights, bias


def init_bn(input_size):
    beta = np.zeros(input_size)
    gamma = np.ones(input_size)
    return beta, gamma


def init_cyclical_lr(cyclical_lr):
    if cyclical_lr is None:
        return None
    eta_min = cyclical_lr[0]
    eta_max = cyclical_lr[1]
    n_s = cyclical_lr[2]
    n_max = cyclical_lr[3] * n_s
    annealing = cyclical_lr[4]
    t = np.linspace(0, 1, n_s)
    up = eta_min + t*(eta_max-eta_min)
    down = eta_max - t*(eta_max-eta_min)
    up_down = up
    for i in range(cyclical_lr[3]):
        if i == 0:
            continue
        elif i % 2 == 1:
            up_down = np.hstack((up_down, down))
        else:
            up_down = np.hstack((up_down, up))
    assert len(up_down) == n_max, 'Length of learning rate array is wrong'
    if annealing is not None:
        n = len(up_down)
        for i, ud in enumerate(up_down):
            up_down[i] = eta_min + (1 - (annealing * i/n)) * (ud - eta_min)
        # plt.plot(up_down)
        # plt.show()
    return up_down, n_max


def ReLU(x):
    return np.maximum(0, x)


def ReLU_prime(x):
    return (x > 0) * 1


def softmax(x):
    """ Standard definition of the softmax function """
    # clip to avoid overflow for float64
    x = np.clip(x, a_min=-300, a_max=300)
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), axis=1)


def cross_entropy(y_pred, one_hot_labels):
    # clip to avoid log(0)
    y_pred = np.clip(y_pred, a_min=1e-100, a_max=np.inf)
    n_samples = one_hot_labels.shape[0]
    ce_loss = np.sum(- one_hot_labels * (np.log(y_pred))) / n_samples
    return ce_loss


def data_augmentation(imgs: np.array, p):
    for i, img in enumerate(imgs):
        img = img.reshape(32, 32, 3, order='F')
        img = np.fliplr(img) if np.random.rand() < p else img
        img = np.flipud(img) if np.random.rand() < p else img
        img = np.roll(img, np.random.randint(-3, 4), axis=0) if np.random.rand() < p else img
        img = np.roll(img, np.random.randint(-3, 4), axis=1) if np.random.rand() < p else img
        imgs[i] = img.reshape(3072, order='F')
    return imgs


def sigmoid(x):
    x = np.clip(x, a_min=0, a_max=300)
    return 1 / (1 + np.exp(-x))


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
