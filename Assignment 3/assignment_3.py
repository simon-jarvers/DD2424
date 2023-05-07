import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


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


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_pass(self, input_data):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_pass(self, output_error, learning_rate, reg_factor):
        raise NotImplementedError

    def get_weight_cost(self, reg_factor):
        return 0


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, sig=None, adam: bool = False):
        super().__init__()
        self.weights, self.bias = init_weights(input_size, output_size, sig)
        self.adam = adam
        if self.adam:
            self.t = 0
            self.m_w = np.zeros_like(self.weights)
            self.m_b = np.zeros_like(self.bias)
            self.v_w = np.zeros_like(self.weights)
            self.v_b = np.zeros_like(self.bias)

    def __repr__(self):
        return f'\n{self.__class__.__name__}: [{self.weights.shape[0]}] --> [{self.weights.shape[1]}]'

    # returns output for a given input
    def forward_pass(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_pass(self, output_error, learning_rate, reg_factor):
        input_error, weights_error, bias_error = self.compute_gradients(output_error, reg_factor)

        # update parameters
        if self.adam:
            self.adam_update(weights_error, bias_error, learning_rate)
        else:
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * bias_error
        return input_error

    def compute_gradients(self, output_error, reg_factor):
        N = output_error.shape[0]
        weights_error = np.dot(self.input.T, output_error) / N
        weights_error += 2 * reg_factor * self.weights
        bias_error = np.expand_dims(output_error.sum(axis=0), axis=0) / N
        # propagate error
        input_error = np.dot(output_error, self.weights.T)
        return input_error, weights_error, bias_error

    def adam_update(self, weights_error, bias_error, learning_rate,
                    beta1: float = 0.9, beta2: float = 0.999,  eps: float = 1e-8):
        self.t += 1
        self.m_w = beta1 * self.m_w + (1 - beta1) * weights_error
        self.m_b = beta1 * self.m_b + (1 - beta1) * bias_error
        m_w_corrected = self.m_w / (1 - beta1 ** self.t)
        m_b_corrected = self.m_b / (1 - beta1 ** self.t)
        self.v_w = beta2 * self.v_w + (1 - beta2) * (weights_error ** 2)
        self.v_b = beta2 * self.v_b + (1 - beta2) * (bias_error ** 2)
        v_w_corrected = self.v_w / (1 - beta2 ** self.t)
        v_b_corrected = self.v_b / (1 - beta2 ** self.t)
        self.weights -= learning_rate * (m_w_corrected / (np.sqrt(v_w_corrected) + eps))
        self.bias -= learning_rate * (m_b_corrected / (np.sqrt(v_b_corrected) + eps))

    def get_weight_cost(self, reg_factor):
        return reg_factor * np.sum(self.weights ** 2)


# inherit from base class Layer
class BNLayer(Layer):
    def __init__(self, input_size, aggregation_mode: str = 'ema'):
        super().__init__()
        self.bn_param = {'mode': 'train', 'eps': 1e-5, 'momentum': 0.9, 'aggregation_mode': aggregation_mode,
                         'running_mean': np.zeros(input_size), 'running_var': np.zeros(input_size)}
        self.sample_means, self.sample_vars = [], []
        self.beta, self.gamma = init_bn(input_size)
        self.input_norm = None
        self.input_centered = None
        self.std = None

    def __repr__(self):
        return f'\n{self.__class__.__name__}: [{self.beta.size}] --> [{self.beta.size}]'

    def set_mode(self, mode):
        self.bn_param['mode'] = mode

    def forward_pass(self, input_data):
        mode = self.bn_param['mode']
        aggregation_mode = self.bn_param['aggregation_mode']
        eps = self.bn_param.get('eps', 1e-5)
        momentum = self.bn_param.get('momentum', 0.9)
        N, D = input_data.shape
        running_mean = self.bn_param.get('running_mean', np.zeros(D, dtype=input_data.dtype))
        running_var = self.bn_param.get('running_var', np.zeros(D, dtype=input_data.dtype))
        self.input = input_data
        if mode == 'train':
            sample_mean = input_data.mean(axis=0)
            sample_var = input_data.var(axis=0)

            if aggregation_mode == 'ema':
                running_mean = momentum * running_mean + (1 - momentum) * sample_mean if running_mean.any() else sample_mean
                running_var = momentum * running_var + (1 - momentum) * sample_var if running_var.any() else sample_var

            elif aggregation_mode == 'pbn':
                self.sample_means.append(sample_mean)
                self.sample_vars.append(sample_var)
                if len(self.sample_means) >= 10:
                    running_mean = np.mean(self.sample_means, axis=0)
                    running_var = np.mean(self.sample_vars, axis=0)
                    self.sample_means = []
                    self.sample_vars = []

            elif aggregation_mode == 'abn':
                running_mean = sample_mean
                running_var = sample_var

            else:
                raise ValueError(f'Invalid forward batchnorm aggregation mode {aggregation_mode}')

            self.std = np.sqrt(sample_var + eps)
            self.input_centered = input_data - sample_mean
            self.input_norm = self.input_centered / self.std
            self.output = self.gamma * self.input_norm + self.beta

        elif mode == 'test':
            x_norm = (input_data - running_mean) / np.sqrt(running_var + eps)
            self.output = self.gamma * x_norm + self.beta

        else:
            raise ValueError(f'Invalid forward batchnorm mode {mode}')

        # Store the updated running means back into bn_param
        self.bn_param['running_mean'] = running_mean
        self.bn_param['running_var'] = running_var

        return self.output

    def backward_pass(self, output_error, learning_rate, reg_factor):
        input_error, beta_error, gamma_error = self.compute_gradients(output_error)

        self.beta -= learning_rate * beta_error
        self.gamma -= learning_rate * gamma_error

        return input_error

    def compute_gradients(self, output_error):
        N = output_error.shape[0]
        gamma_error = (output_error * self.input_norm).sum(axis=0) / N
        beta_error = output_error.sum(axis=0) / N

        x_norm_error = output_error * self.gamma
        x_centered_error = x_norm_error / self.std
        mean_error = -(x_centered_error.sum(axis=0) + 2 / N * self.input_centered.sum(axis=0))
        std_error = (x_norm_error * self.input_centered * -self.std ** (-2)).sum(axis=0)
        var_error = std_error / 2 / self.std
        input_error = x_centered_error + (mean_error + var_error * 2 * self.input_centered) / N

        return input_error, beta_error, gamma_error


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation=ReLU, activation_prime=ReLU_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def __repr__(self):
        return f'\n{self.__class__.__name__}: {self.activation.__name__}'

    # returns the activated input
    def forward_pass(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_pass(self, output_error, learning_rate, reg_factor):
        return self.activation_prime(self.input) * output_error


class ClassificationNetwork:
    def __init__(self, hidden_sizes=None, reg_factor=0.0, n_batch=100, eta=.001,
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


def main():
    class_net = ClassificationNetwork(reg_factor=0.005, n_epochs=50, n_batch=100,
                                      # hidden_sizes=[50, 50],
                                      hidden_sizes=[512, 256, 128, 64, 32, 16],
                                      n_train_samples=45000, n_val_samples=5000,
                                      cyclical_lr=[1e-5, 1e-1, 2250, 4, 0],
                                      # eta=0.001,
                                      # after_n_epochs=10,
                                      # adam=True,
                                      augment_p=0.0,
                                      # weight_sig=1e-1,
                                      bn_aggregation_mode='ema'
                                      )

    print(class_net)
    class_net.train()
    # check_gradients(class_net)
    # class_net.plot_histogram()


def check_gradients(class_net):
    n_samples = 5
    red_dim = 10
    class_net.reduce_dim(red_dim)
    data = class_net.data['train']['data'][:n_samples, :]
    labels = class_net.data['train']['one_hot_labels'][:n_samples, :]

    grads_num = class_net.compute_gradients_num(data, labels, h=1e-5)
    grads_ana = class_net.compute_gradients(data, labels)
    names = [name for item in reversed([[f'W{i+1}', f'b{i+1}'] for i in range(int(len(grads_ana)/2))]) for name in item]
    for i, (grad_ana, grad_num) in enumerate(zip(grads_ana, grads_num)):
        rel_diffs = []
        abs_diffs = []
        for ana, num in zip(grad_ana, grad_num):
            for a, n in zip(ana, num):
                rel_diff = np.abs(a - n)/max(np.float64(1e-1), (np.abs(a) + np.abs(n)))
                rel_diffs.append(rel_diff)
                abs_diff = np.abs(a - n)
                abs_diffs.append(abs_diff)

        fig, ax = plt.subplots(1, 2, figsize=(7, 12))
        ax[0].boxplot(rel_diffs)
        ax[0].set_title(f'Relative differences for {names[i]}')
        ax[1].boxplot(abs_diffs)
        ax[1].set_title(f'Absolute differences for {names[i]}')
        plt.savefig(f'./figs/after_BN/{names[i]}')


def grid_search():
    n_epochs = 100
    cyclical_lr = [1e-5, 1e-1, 2250, 4, 0.8]
    n_batch = 100
    reg_factors = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    hidden_sizes = [50, 50]
    augmentation_p = 0.0

    for reg_factor in reg_factors:
        class_net = ClassificationNetwork(reg_factor=reg_factor, n_batch=n_batch, n_epochs=n_epochs,
                                          hidden_sizes=hidden_sizes,
                                          n_train_samples=45000, n_val_samples=5000,
                                          cyclical_lr=cyclical_lr,
                                          augment_p=augmentation_p,
                                          silent=True)
        print(class_net)
        val_acc = class_net.train()
        print(f'Final validation accuracy: {100*val_acc:.2f}%')


if __name__ == '__main__':
    main()


