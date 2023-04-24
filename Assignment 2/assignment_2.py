import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    one_hot_labels = np.zeros((n_labels, labels.size))
    np.put_along_axis(one_hot_labels, np.expand_dims(labels, axis=0), 1, axis=0)
    if first_n_samples is not None and last_n_samples is not None:
        train_split = {'data': data[:first_n_samples, :],
                       'labels': labels[:first_n_samples],
                       'one_hot_labels': one_hot_labels[:, :first_n_samples]}
        val_split = {'data': data[-last_n_samples:, :],
                     'labels': labels[-last_n_samples:],
                     'one_hot_labels': one_hot_labels[:, -last_n_samples:]}
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


def init_weights(input_size: int, output_size: int) -> tuple:
    weights = np.random.normal(0, 1/np.sqrt(input_size), (output_size, input_size))
    bias = np.random.normal(0, 1/np.sqrt(input_size), (output_size, 1))
    return weights, bias


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


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    """ Standard definition of the softmax function """
    # clip to avoid overflow for float64
    # x = np.clip(x, a_min=0, a_max=300)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(y_pred, one_hot_labels):
    # clip to avoid log(0)
    y_pred = np.clip(y_pred, a_min=1e-100, a_max=np.inf)
    n_samples = one_hot_labels.shape[1]
    ce_loss = np.sum(- one_hot_labels * (np.log(y_pred))) / n_samples
    return ce_loss


def data_augmentation(imgs: np.array, p):
    for i, img in enumerate(imgs):
        img = img.reshape(32, 32, 3, order='F')
        img = np.fliplr(img) if np.random.rand() < p else img
        img = np.flipud(img) if np.random.rand() < p else img
        img = np.roll(img, np.random.randint(-3, 4), axis=0) if np.random.rand() < p else img
        img = np.roll(img, np.random.randint(-3, 4), axis=1) if np.random.rand() < p else img
        imgs[i] = np.reshape(img, 3072)
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


def eval_2_layer(X, W1, b1, W2, b2) -> np.array:
    s1 = W1.dot(X.T) + b1
    h = relu(s1)
    s2 = W2.dot(h) + b2
    z = softmax(s2)
    return z


def cost_2_layer(X, Y, W1, b1, W2, b2, reg_factor) -> np.array:
    y_pred = eval_2_layer(X, W1, b1, W2, b2)
    pred_loss = cross_entropy(y_pred, Y)
    weight_decay = reg_factor * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    return pred_loss + weight_decay


class ClassificationNetwork:
    def __init__(self, hidden_size: int = 50, reg_factor=0.0, n_batch=100, eta=.001,
                 cyclical_lr=None, n_epochs=20, after_n_epochs=None, silent=False,
                 n_train_samples=40000,
                 n_val_samples=10000,
                 augment_p=.01,
                 mu=0.0):
        self.silent = silent

        # hyperparameter definition
        self.hidden_size = hidden_size
        self.reg_factor = reg_factor
        self.n_batch = n_batch
        self.eta = eta
        if cyclical_lr is not None:
            self.cyclical_lr, self.n_s_max = init_cyclical_lr(cyclical_lr)
        else:
            self.cyclical_lr = cyclical_lr
            self.n_s_max = int(n_epochs * n_train_samples / n_batch)
        self.n_epochs = n_epochs
        self.after_n_epochs = after_n_epochs
        self.augment_p = augment_p
        self.mu = mu

        # data class names
        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # load data
        if n_train_samples is not None and n_val_samples is not None:
            assert n_train_samples + n_val_samples <= 50000, 'Training and Validation Set overlap.'
        train_data, val_data = load_images_and_labels([f'./cifar-10-batches-py/data_batch_1',
                                                      f'./cifar-10-batches-py/data_batch_3',
                                                      f'./cifar-10-batches-py/data_batch_4',
                                                      f'./cifar-10-batches-py/data_batch_5',
                                                      f'./cifar-10-batches-py/data_batch_2'],
                                                      n_train_samples, n_val_samples)
        # valid_split = np.in1d(train_data['data'], val_data['data']).any()
        # assert valid_split, 'Training and Validation Set overlap.'
        test_data = load_images_and_labels(f'./cifar-10-batches-py/test_batch')
        self.data = {'train': train_data,
                     'val': val_data,
                     'test': test_data}
        self.K = 10
        self.d = self.data['train']['data'][0].size
        self.n_train = self.data['train']['data'].shape[0]

        # normalize data
        self.normalize_data()

        # init weights and bias
        self.w1, self.b1 = init_weights(self.d, self.hidden_size)
        self.w2, self.b2 = init_weights(self.hidden_size, self.K)

        # init momentum
        self.w1_momentum, self.b1_momentum = np.zeros_like(self.w1), np.zeros_like(self.b1)
        self.w2_momentum, self.b2_momentum = np.zeros_like(self.w2), np.zeros_like(self.b2)

    def __repr__(self):
        return f'\nSettings:\nNumber of hidden units: {self.hidden_size}\nRegularization factor: {self.reg_factor}\n' \
               f'Number of training steps: {self.n_s_max}\nData augmentation probability: {self.augment_p}' \
               f'\nMomentum: {self.mu}'

    def normalize_data(self) -> None:
        mean = np.mean(self.data['train']['data'], axis=0)
        std = np.std(self.data['train']['data'], axis=0)
        self.data['train']['data'] = (self.data['train']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]
        self.data['val']['data'] = (self.data['val']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]
        self.data['test']['data'] = (self.data['test']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]

    def eval(self, X) -> np.array:
        s1 = self.w1.dot(X.T) + self.b1
        self.h = relu(s1)
        return eval_2_layer(X, self.w1, self.b1, self.w2, self.b2)

    def cost(self, X, one_hot_labels):
        return cost_2_layer(X, one_hot_labels, self.w1, self.b1, self.w2, self.b2, self.reg_factor)

    def acc(self, X, labels):
        y_pred = self.eval(X)
        labels_pred = np.argmax(y_pred, axis=0)
        acc = np.sum(np.equal(labels, labels_pred)) / len(labels)
        return acc

    def compute_gradients(self, X, one_hot_labels) -> list:
        grad_w1, grad_b1 = np.zeros_like(self.w1), np.zeros_like(self.b1)
        grad_w2, grad_b2 = np.zeros_like(self.w2), np.zeros_like(self.b2)
        n_samples = one_hot_labels.shape[1]
        y_pred = self.eval(X)
        g = (y_pred - one_hot_labels)
        grad_b2 += np.expand_dims(g.sum(axis=1), 1) / n_samples
        grad_w2 += g.dot(self.h.T) / n_samples
        grad_w2 += 2 * self.reg_factor * self.w2
        g = self.w2.T.dot(g) * np.sign(self.h)
        grad_b1 += np.expand_dims(g.sum(axis=1), 1) / n_samples
        grad_w1 += g.dot(X) / n_samples
        grad_w1 += 2 * self.reg_factor * self.w1
        return [grad_w1, grad_b1, grad_w2, grad_b2]

    def gd_step(self, X, one_hot_labels):
        w1_momentum, b1_momentum, w2_momentum, b2_momentum = self.w1_momentum, self.b1_momentum, self.w2_momentum, self.b2_momentum
        self.w1 = self.w1 + self.mu * w1_momentum
        self.b1 = self.b1 + self.mu * b1_momentum
        self.w2 = self.w2 + self.mu * w2_momentum
        self.b2 = self.b2 + self.mu * b2_momentum

        grad_w1, grad_b1, grad_w2, grad_b2 = self.compute_gradients(X, one_hot_labels)
        self.w1_momentum = self.mu * w1_momentum - self.eta * grad_w1
        self.b1_momentum = self.mu * b1_momentum - self.eta * grad_b1
        self.w2_momentum = self.mu * w2_momentum - self.eta * grad_w2
        self.b2_momentum = self.mu * b2_momentum - self.eta * grad_b2

        self.w1 += -self.mu * w1_momentum + (1 + self.mu) * self.w1_momentum
        self.b1 += -self.mu * b1_momentum + (1 + self.mu) * self.b1_momentum
        self.w2 += -self.mu * w2_momentum + (1 + self.mu) * self.w2_momentum
        self.b2 += -self.mu * b2_momentum + (1 + self.mu) * self.b2_momentum

    def train(self):
        train_cost = []
        val_cost = []
        train_acc = []
        val_acc = []
        etas = []
        for epoch in tqdm(range(self.n_epochs), desc='Processing epoch:'):
            if self.after_n_epochs is not None:
                if epoch % self.after_n_epochs == 0 and epoch > 0:
                    self.eta *= 0.5
            n_batches = self.data['train']['data'].shape[0] / self.n_batch
            idx = np.random.permutation(self.n_train)
            shuffled_train_data = self.data['train']['data'][idx]
            shuffled_train_labels = self.data['train']['one_hot_labels'][:, idx]
            train_data = np.array_split(shuffled_train_data, n_batches, axis=0)
            train_labels = np.array_split(shuffled_train_labels, n_batches, axis=1)
            for train_step, (train_X, train_label) in enumerate(zip(train_data, train_labels)):
                if self.cyclical_lr is not None:
                    t = epoch * len(train_data) + train_step
                    if t >= self.n_s_max:
                        break
                    self.eta = self.cyclical_lr[t]
                    etas.append(self.eta)
                if np.random.rand() < self.augment_p:
                    train_X = data_augmentation(train_X, min(10*self.augment_p, 0.5))
                self.gd_step(train_X, train_label)
            else:
                train_cost.append(self.cost(self.data['train']['data'], self.data['train']['one_hot_labels']))
                val_cost.append(self.cost(self.data['val']['data'], self.data['val']['one_hot_labels']))
                train_acc.append(self.acc(self.data['train']['data'], self.data['train']['labels']))
                val_acc.append(self.acc(self.data['val']['data'], self.data['val']['labels']))
                continue
            break

        final_val_acc = self.acc(self.data['val']['data'], self.data['val']['labels'])
        if not self.silent:
            print(f'Final Validation accuracy: {100 * final_val_acc:.2f} %')

            final_test_acc = self.acc(self.data['test']['data'], self.data['test']['labels'])
            print(f'Final Test accuracy: {100 * final_test_acc:.2f} %')

        # cost and accuracy plot
        if not self.silent:
            fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))
            ax1.set_title('Cost over epochs')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Cost')
            ax1.plot(train_cost, label='Training cost')
            ax1.plot(val_cost, label='Validation cost')
            ax1.legend()
            ax2.set_title('Accuracy over epochs')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.plot(train_acc, label='Training accuracy')
            ax2.plot(val_acc, label='Validation accuracy')
            ax2.legend()
            plt.show()
            plt.plot(etas)
            plt.xlabel('Training steps')
            plt.ylabel('Learning rate')
            plt.show()

        return final_val_acc

    def plot_weight_images(self):
        """ Display the image for each label in W """
        fig, ax = plt.subplots(1, 10, figsize=(12, 2))
        for i in range(10):
            img = self.w1[i, :].reshape(32, 32, 3, order='F')
            sim = (img-np.min(img[:]))/(np.max(img[:])-np.min(img[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i].imshow(sim, interpolation='nearest')
            ax[i].set_title(self.classnames[i])
            ax[i].axis('off')
        plt.show()

    def plot_histogram(self):
        labels_pred = np.argmax(self.eval(self.data['test']['data']), axis=0)
        labels_true = self.data['test']['labels']

        # 2d histogram
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Predicted vs true labels')
        ax.hist2d(labels_true, labels_pred)
        ax.set_xlabel('True labels')
        ax.set_xticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        ax.set_ylabel('Predicted labels')
        ax.set_yticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        plt.show()

        # 1d histogram
        correct_labels = []
        incorrect_labels = []
        for label_true, label_pred in zip(labels_true, labels_pred):
            if label_true == label_pred:
                correct_labels.append(label_true)
            else:
                incorrect_labels.append(label_true)

        # Creating histogram
        fig, ax = plt.subplots(2, 1, figsize=(10, 14))
        ax[0].hist(correct_labels, color='green')
        ax[0].set_title('Correctly classified labels')
        ax[0].set_xticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        ax[1].hist(incorrect_labels, color='red')
        ax[1].set_title('Incorrectly classified labels')
        ax[1].set_xticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        plt.show()

    def compute_gradients_num(self, X, Y, h):
        """ Converted from matlab code """

        grad_W1 = np.zeros(self.w1.shape)
        grad_b1 = np.zeros(self.b1.shape)
        grad_W2 = np.zeros(self.w2.shape)
        grad_b2 = np.zeros(self.b2.shape)

        c = self.cost(X, Y)

        for i in range(len(self.b1)):
            b_try = np.array(self.b1)
            b_try[i] += h
            c2 = cost_2_layer(X, Y, self.w1, b_try, self.w2, self.b2, self.reg_factor)
            grad_b1[i] = (c2 - c) / h

        for i in range(len(self.b2)):
            b_try = np.array(self.b2)
            b_try[i] += h
            c2 = cost_2_layer(X, Y, self.w1, self.b1, self.w2, b_try, self.reg_factor)
            grad_b2[i] = (c2 - c) / h

        for i in tqdm(range(self.w2.shape[0])):
            for j in range(self.w2.shape[1]):
                W_try = np.array(self.w2)
                W_try[i, j] += h
                c2 = cost_2_layer(X, Y, self.w1, self.b1, W_try, self.b2, self.reg_factor)
                grad_W2[i, j] = (c2 - c) / h

        for i in tqdm(range(self.w1.shape[0])):
            for j in range(self.w1.shape[1]):
                W_try = np.array(self.w1)
                W_try[i, j] += h
                c2 = cost_2_layer(X, Y, W_try, self.b1, self.w2, self.b2, self.reg_factor)
                grad_W1[i, j] = (c2-c) / h

        return [grad_W1, grad_b1, grad_W2, grad_b2]


def main():
    class_net = ClassificationNetwork(reg_factor=0.001, n_epochs=100, n_batch=100,
                                      hidden_size=200,
                                      n_train_samples=49000, n_val_samples=1000,
                                      cyclical_lr=[1e-5, 1e-1, 1000, 6, 1],
                                      # eta=0.01, after_n_epochs=5,
                                      augment_p=.05,
                                      mu=0.5)

    print(class_net)

    # check_gradients(class_net)

    class_net.train()
    class_net.plot_histogram()
    class_net.plot_weight_images()


def check_gradients(class_net):
    n_samples = 1
    data = class_net.data['train']['data'][:n_samples, :]
    labels = class_net.data['train']['one_hot_labels'][:, :n_samples]

    grads_ana = class_net.compute_gradients(data, labels)
    grads_num = class_net.compute_gradients_num(data, labels, h=0.00001)
    names = ['W1', 'b1', 'W2', 'b2']
    for i, (grad_ana, grad_num) in enumerate(zip(grads_ana, grads_num)):
        rel_diffs = []
        abs_diffs = []
        for ana, num in zip(grad_ana, grad_num):
            for a, n in zip(ana, num):
                rel_diff = np.abs(a - n)/max(np.float64(0.000001), (np.abs(a) + np.abs(n)))
                rel_diffs.append(rel_diff)
                abs_diff = np.abs(a - n)
                abs_diffs.append(abs_diff)

        fig, ax = plt.subplots(1, 2, figsize=(7, 12))
        ax[0].boxplot(rel_diffs)
        ax[0].set_title(f'Relative differences for {names[i]}')
        ax[1].boxplot(abs_diffs)
        ax[1].set_title(f'Absolute differences for {names[i]}')
        plt.show()


def grid_search():
    n_epochs = 100
    cyclical_lr = [1e-5, 1e-1, 1000, 0, 0.8]
    n_cycles = [6, 8]
    n_batch = 100
    log_reg_factors = [-4, -3]
    hidden_sizes = [100, 400]
    augmentation_p = 0.05
    mu = 0.5

    for log_reg_factor in log_reg_factors:
        reg_factor = 10**log_reg_factor
        for n_cycle in n_cycles:
            cyclical_lr[3] = n_cycle
            for hidden_size in hidden_sizes:

                class_net = ClassificationNetwork(reg_factor=reg_factor, n_batch=n_batch, n_epochs=n_epochs,
                                                  hidden_size=hidden_size,
                                                  n_train_samples=49000, n_val_samples=1000,
                                                  cyclical_lr=cyclical_lr,
                                                  augment_p=augmentation_p,
                                                  mu=mu,
                                                  silent=True)
                print(class_net)
                val_acc = class_net.train()
                print(f'Final validation accuracy: {100*val_acc:.2f}%')


if __name__ == '__main__':
    main()

