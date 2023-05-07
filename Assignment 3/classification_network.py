import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from utils import load_images_and_labels, init_weights, init_cyclical_lr, softmax, data_augmentation, cross_entropy
from layer import Layer, FCLayer, ActivationLayer, BNLayer


class ClassificationNetwork:
    def __init__(self, hidden_sizes=None, reg_factor=0, n_batch=100, eta=.001,
                 cyclical_lr=None, n_epochs=20, after_n_epochs=None, silent=False,
                 n_train_samples=40000,
                 n_val_samples=10000,
                 augment_p=.0,
                 weight_sig=None,
                 bn_aggregation_mode='ema',
                 adam=False):
        self.silent = silent

        # hyperparameter definition
        self.hidden_sizes = hidden_sizes
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
        self.bn_aggregation_mode = bn_aggregation_mode

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
        self.N, self.D = self.data['train']['data'].shape

        # normalize data
        self.normalize_data()

        # init layers
        self.layers = []
        if self.hidden_sizes is None:
            self.add_layer(FCLayer(self.D, self.K))
        else:
            for i, hidden_size in enumerate(self.hidden_sizes):
                if i == 0:
                    self.add_layer(FCLayer(self.D, self.hidden_sizes[0], sig=weight_sig, adam=adam))
                    self.add_layer(BNLayer(self.hidden_sizes[0], aggregation_mode=bn_aggregation_mode))
                    self.add_layer(ActivationLayer())
                else:
                    self.add_layer(FCLayer(self.hidden_sizes[i-1], hidden_size, sig=weight_sig, adam=adam))
                    self.add_layer(BNLayer(self.hidden_sizes[i], aggregation_mode=bn_aggregation_mode))
                    self.add_layer(ActivationLayer())
            self.add_layer(FCLayer(self.hidden_sizes[-1], self.K, sig=weight_sig))

    def __repr__(self):
        return f'\nSettings:\nNumber of hidden units: {self.hidden_sizes}\nRegularization factor: {self.reg_factor}\n' \
               f'Number of training steps: {self.n_s_max}\nData augmentation probability: {self.augment_p} \n' \
               f'Architecture: {self.layers}'

    def add_layer(self, layer):
        self.layers.append(layer)

    def normalize_data(self) -> None:
        mean = np.mean(self.data['train']['data'], axis=0)
        std = np.std(self.data['train']['data'], axis=0)
        self.data['train']['data'] = (self.data['train']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]
        self.data['val']['data'] = (self.data['val']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]
        self.data['test']['data'] = (self.data['test']['data'] - mean[np.newaxis, :]) / std[np.newaxis, :]

    def reduce_dim(self, red_dim):
        self.data['train']['data'] = self.data['train']['data'][:, :red_dim]
        self.data['val']['data'] = self.data['val']['data'][:, :red_dim]
        self.data['test']['data'] = self.data['test']['data'][:, :red_dim]
        self.layers[0].weights = self.layers[0].weights[:red_dim, :]
        self.layers[0].bias = self.layers[0].bias[:red_dim, :]

    def eval(self, X) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)
        return softmax(output)

    def cost(self, X, one_hot_labels):
        y_pred = self.eval(X)
        pred_loss = cross_entropy(y_pred, one_hot_labels)
        weight_lost = np.sum([layer.get_weight_cost(self.reg_factor) for layer in self.layers])
        return pred_loss + weight_lost

    def acc(self, X, labels):
        y_pred = self.eval(X)
        labels_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(np.equal(labels, labels_pred)) / len(labels)
        return acc

    def compute_gradients(self, X, one_hot_labels) -> list:
        y_pred = self.eval(X)
        error = (y_pred - one_hot_labels)
        grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, FCLayer):
                # layer.backward_pass(error, self.eta, self.reg_factor)
                error, grad_w, grad_b = layer.compute_gradients(error, self.reg_factor)
                grads.append(grad_w)
                grads.append(grad_b)
            elif isinstance(layer, BNLayer):
                # layer.backward_pass(error, self.eta, self.reg_factor)
                error, _, _ = layer.compute_gradients(error)
            elif isinstance(layer, ActivationLayer):
                error = layer.backward_pass(error, self.eta, self.reg_factor)
            else:
                raise ValueError(f'Invalid layer {layer}')
        return grads

    def gd_step(self, X, one_hot_labels):
        y_pred = self.eval(X)
        error = (y_pred - one_hot_labels)
        for layer in reversed(self.layers):
            error = layer.backward_pass(error, self.eta, self.reg_factor)

    def set_bn_mode(self, mode):
        for layer in self.layers:
            if isinstance(layer, BNLayer):
                layer.set_mode(mode)

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
            idx = np.random.permutation(self.N)
            shuffled_train_data = self.data['train']['data'][idx]
            shuffled_train_labels = self.data['train']['one_hot_labels'][idx, :]
            train_data = np.array_split(shuffled_train_data, n_batches, axis=0)
            train_labels = np.array_split(shuffled_train_labels, n_batches, axis=0)
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

        if self.bn_aggregation_mode == 'abn':
            augmented_val_data = data_augmentation(self.data['val']['data'], p=0)
            self.eval(augmented_val_data)

        self.set_bn_mode('test')
        final_val_acc = self.acc(self.data['val']['data'], self.data['val']['labels'])

        if not self.silent:
            print(f'Final Validation accuracy: {100 * final_val_acc:.2f} %')

            self.set_bn_mode('train')
            if self.bn_aggregation_mode == 'abn':
                augmented_test_data = data_augmentation(self.data['test']['data'], p=0)
                self.eval(augmented_test_data)
            self.set_bn_mode('test')

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

        fc_layers = list(filter(lambda l: isinstance(l, FCLayer), self.layers))
        grads = []

        for layer in reversed(fc_layers):

            layer_idx = self.layers.index(layer)

            grad_W = np.zeros(layer.weights.shape)
            grad_b = np.zeros(layer.bias.shape)

            c = self.cost(X, Y)

            for i in range(layer.bias.shape[1]):
                b_try = np.array(layer.bias)
                b_try[0, i] += h
                temp_net = deepcopy(self)
                temp_net.layers[layer_idx].bias = b_try
                c2 = temp_net.cost(X, Y)
                grad_b[0, i] = (c2 - c) / h

            for i in tqdm(range(layer.weights.shape[0])):
                for j in range(layer.weights.shape[1]):
                    W_try = np.array(layer.weights)
                    W_try[i, j] += h
                    temp_net = deepcopy(self)
                    temp_net.layers[layer_idx].weights = W_try
                    c2 = temp_net.cost(X, Y)
                    grad_W[i, j] = (c2-c) / h

            grads.append(grad_W)
            grads.append(grad_b)

        return grads




