import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    weights = np.random.normal(0, 0.01, (K, d))
    bias = np.random.normal(0, 0.01, (K, 1))
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


class ClassificationNetwork:
    def __init__(self, loss_function: str = 'ce', reg_factor=0.0, n_batch=100, eta=.001, n_epochs=20,
                 after_n_epochs=None, silent=False):
        self.silent = silent
        assert loss_function in ['ce', 'bce'], 'please chose "ce" or "bce" as loss function'
        self.loss_function = loss_function

        # hyperparameter definition
        self.reg_factor = reg_factor
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs
        self.after_n_epochs = after_n_epochs

        # data class names
        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # load data
        self.data = {'train': load_images_and_labels([f'./cifar-10-batches-py/data_batch_1',
                                                      f'./cifar-10-batches-py/data_batch_3',
                                                      f'./cifar-10-batches-py/data_batch_4',
                                                      f'./cifar-10-batches-py/data_batch_5']),
                     'val': load_images_and_labels(f'./cifar-10-batches-py/data_batch_2'),
                     'test': load_images_and_labels(f'./cifar-10-batches-py/test_batch')}
        self.K = 10
        self.d = self.data['train']['data'][0].size
        self.n_train = self.data['train']['data'].shape[0]

        # normalize data
        self.normalize_data()

        # init weights and bias
        self.weights, self.bias = init_weights(self.K, self.d)

    def normalize_data(self) -> None:
        mean = np.mean(self.data['train']['data'], axis=0)
        std = np.std(self.data['train']['data'], axis=0)
        self.data['train']['data'] = (self.data['train']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]
        self.data['val']['data'] = (self.data['val']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]
        self.data['test']['data'] = (self.data['test']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]

    def eval(self, X) -> np.array:
        s = self.weights.dot(X.T) + self.bias
        if self.loss_function == 'bce':
            z = sigmoid(s)
        else:
            z = softmax(s)
        return z

    def cost(self, X, one_hot_labels):
        y_pred = self.eval(X)
        if self.loss_function == 'bce':
            pred_loss = binary_cross_entropy(y_pred, one_hot_labels)
        else:
            pred_loss = cross_entropy(y_pred, one_hot_labels)
        weight_decay = self.reg_factor * np.sum(self.weights**2)
        return pred_loss + weight_decay

    def acc(self, X, labels):
        y_pred = self.eval(X)
        labels_pred = np.argmax(y_pred, axis=0)
        acc = np.sum(np.equal(labels, labels_pred)) / len(labels)
        return acc

    def compute_gradients(self, X, one_hot_labels) -> tuple:
        grad_weights, grad_bias = np.zeros_like(self.weights), np.zeros_like(self.bias)
        n_samples = one_hot_labels.shape[1]
        y_pred = self.eval(X)
        g = -(one_hot_labels - y_pred)
        if self.loss_function == 'bce':
            grad_bias += np.expand_dims(g.sum(axis=1), 1) / (n_samples * self.K)
            grad_weights += g.dot(X) / (n_samples * self.K)
            grad_weights += 2 * self.reg_factor * self.weights
        else:
            grad_bias += np.expand_dims(g.sum(axis=1), 1) / n_samples
            grad_weights += g.dot(X) / n_samples
            grad_weights += 2 * self.reg_factor * self.weights
        return grad_weights, grad_bias

    def gd_step(self, X, one_hot_labels):
        grad_weights, grad_bias = self.compute_gradients(X, one_hot_labels)
        self.weights -= self.eta * grad_weights
        self.bias -= self.eta * grad_bias

    def train(self):
        train_cost = []
        val_cost = []
        train_acc = []
        val_acc = []
        for epoch in tqdm(range(self.n_epochs), desc='Processing epoch:', disable=self.silent):
            if self.after_n_epochs is not None:
                if epoch % self.after_n_epochs == 0 and epoch > 0:
                    self.eta *= 0.5
                    # if not self.silent:
                    #     print(f'\nAfter {epoch} epochs decreasing the learning rate to : {self.eta}')
            n_batches = self.data['train']['data'].shape[0] / self.n_batch
            idx = np.random.permutation(self.n_train)
            shuffled_train_data = self.data['train']['data'][idx]
            shuffled_train_labels = self.data['train']['one_hot_labels'][:, idx]
            train_data = np.array_split(shuffled_train_data, n_batches, axis=0)
            train_labels = np.array_split(shuffled_train_labels, n_batches, axis=1)
            for train_X, train_label in zip(train_data, train_labels):
                self.gd_step(train_X, train_label)
            train_cost.append(self.cost(self.data['train']['data'], self.data['train']['one_hot_labels']))
            val_cost.append(self.cost(self.data['val']['data'], self.data['val']['one_hot_labels']))
            train_acc.append(self.acc(self.data['train']['data'], self.data['train']['labels']))
            val_acc.append(self.acc(self.data['val']['data'], self.data['val']['labels']))

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

        final_val_acc = self.acc(self.data['val']['data'], self.data['val']['labels'])
        if not self.silent:
            print(f'Final Validation accuracy: {100*final_val_acc:.2f} %')

            final_test_acc = self.acc(self.data['test']['data'], self.data['test']['labels'])
            print(f'Final Test accuracy: {100*final_test_acc:.2f} %')
        return final_val_acc

    def plot_weight_images(self):
        """ Display the image for each label in W """
        fig, ax = plt.subplots(1, 10, figsize=(12, 2))
        for i in range(10):
            img = self.weights[i, :].reshape(32, 32, 3, order='F')
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
        fig, ax = plt.subplots(2, 1, figsize=(10, 14))
        ax[0].hist(correct_labels, color='green')
        ax[0].set_title('Correctly classified labels')
        ax[0].set_xticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        ax[1].hist(incorrect_labels, color='red')
        ax[1].set_title('Incorrectly classified labels')
        ax[1].set_xticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        plt.show()


def grid_search():
    n_epochs = 40
    after_n_epochs = [5, 10, 20]
    etas = [.1, .01, .001]
    n_batchs = [50, 100, 500]
    reg_factors = [1, .1, .01]

    for after_n_epoch, eta in zip(after_n_epochs, etas):
        for n_batch in n_batchs:
            for reg_factor in reg_factors:
                class_net = ClassificationNetwork(reg_factor=reg_factor, n_batch=n_batch, eta=eta, n_epochs=n_epochs,
                                                  after_n_epochs=after_n_epoch, silent=True)
                val_acc = class_net.train()
                print(f'For Settings: reg_factor={reg_factor}, n_batch={n_batch}, eta={eta}, n_epochs={n_epochs},'
                      f'after_n_epochs= {after_n_epoch} the final validation accuracy is {val_acc}%')


def main():

    class_net = ClassificationNetwork(loss_function='bce', reg_factor=0.1, n_epochs=40, eta=.001, n_batch=100, after_n_epochs=10)

    # n_samples = 10
    #
    # grad_W_ana, grad_b_ana = class_net.compute_gradients(class_net.data['train']['data'][:n_samples, :],
    #                             class_net.data['train']['one_hot_labels'][:, :n_samples])
    # grad_W_num, grad_b_num = ComputeGradsNum(class_net.data['train']['data'][:n_samples, :],
    #                                          class_net.data['train']['one_hot_labels'][:, :n_samples],
    #                                          0,
    #                                          class_net.weights,
    #                                          class_net.bias,
    #                                          class_net.reg_factor,
    #                                          h=0.00001)
    # max_rel_diff = 0
    # max_abs_diff = 0
    # for ana, num in zip(grad_b_ana, grad_b_num):
    #     for a, n in zip(ana, num):
    #         rel_diff = np.abs(a - n)/max(np.float64(0.000001), (np.abs(a) + np.abs(n)))
    #         max_rel_diff = rel_diff if rel_diff > max_rel_diff else max_rel_diff
    #         abs_diff = a - n
    #         max_abs_diff = abs_diff if abs_diff > max_abs_diff else max_abs_diff
    # print(f'Maximal relative gradient difference: {max_rel_diff}')
    # print(f'Maximal absolute gradient difference: {max_abs_diff}')

    class_net.train()
    class_net.plot_histogram()
    class_net.plot_weight_images()


if __name__ == '__main__':
    main()
