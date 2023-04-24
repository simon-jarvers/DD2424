import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_images_and_labels, init_weights, softmax, cross_entropy, binary_cross_entropy, sigmoid


class ClassificationNetwork:
    def __init__(self, loss_function: str = 'ce', reg_factor=0, n_batch=100, eta=.001, n_epochs=20,
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

        # Creating histogram
        fig, ax = plt.subplots(2, 1, figsize=(10, 14))
        ax[0].hist(correct_labels, color='green')
        ax[0].set_title('Correctly classified labels')
        ax[0].set_xticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        ax[1].hist(incorrect_labels, color='red')
        ax[1].set_title('Incorrectly classified labels')
        ax[1].set_xticks([0.5+k*0.9 for k in range(self.K)], self.classnames)
        plt.show()




