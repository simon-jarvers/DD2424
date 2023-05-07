import numpy as np
from utils import init_weights, softmax, ReLU, ReLU_prime, init_bn


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
