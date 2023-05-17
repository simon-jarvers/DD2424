import matplotlib.pyplot as plt
import numpy as np

from utils import get_book_data, softmax, save_text
from copy import deepcopy
from tqdm import tqdm
import pickle


class DataLoader:
    def __init__(self, book_fname, seq_length):
        self.book_text: str = get_book_data(book_fname)
        self.data = list(self.book_text)
        self.input_size, self.ind_to_char, self.char_to_ind = self.init_mapping()
        self.seq_length = seq_length
        self.pointer = 0

    def init_mapping(self):
        unique_chars = list(set(self.book_text))
        unique_chars.sort()
        char_to_ind = {value: key for value, key in enumerate(unique_chars)}
        ind_to_char = {key: value for value, key in enumerate(unique_chars)}
        return len(unique_chars), char_to_ind, ind_to_char

    def char_to_one_hot(self, char: str) -> np.array:
        idx = self.char_to_ind[char]
        vec = np.zeros(self.input_size)
        vec[idx] = 1
        return vec

    def one_hot_to_char(self, one_hot: np.array) -> str:
        idx = np.argmax(one_hot)
        char = self.ind_to_char[idx]
        return char

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ind[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ind[ch] for ch in self.data[input_start + 1:input_end + 1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= len(self.data):
            # reset pointer
            self.pointer = 0
        return inputs, targets


class RNN:
    def __init__(self, book_fname, hidden_size=100, eta=.1, seq_length=25, sig=.01, n_iterations=100000):
        self.dataloader = DataLoader(book_fname, seq_length)
        self.input_size = self.dataloader.input_size
        self.hidden_size = hidden_size
        self.eta = eta
        self.seq_length = seq_length
        self.n_iterations = n_iterations

        # init training
        self.smooth_loss = [-np.log(1.0 / self.input_size) * self.seq_length]
        self.hprev = np.zeros(self.hidden_size)

        # init weights
        self.b = np.zeros(self.hidden_size)
        self.c = np.zeros(self.input_size)
        self.U = np.random.randn(self.hidden_size, self.input_size) * sig
        self.W = np.random.randn(self.hidden_size, self.hidden_size) * sig
        self.V = np.random.randn(self.input_size, self.hidden_size) * sig

        # adagrad variables
        self.mU = np.zeros_like(self.U)
        self.mW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)

    def forward(self, inputs, hprev):
        xs, hs, os, ycap = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        for t, input in enumerate(inputs):
            xs[t] = self.dataloader.char_to_one_hot(self.dataloader.ind_to_char[input])
            hs[t] = np.tanh(np.dot(self.U, xs[t]) + np.dot(self.W, hs[t - 1]) + self.b)  # hidden state
            os[t] = np.dot(self.V, hs[t]) + self.c  # unnormalised log probs for next char
            ycap[t] = softmax(os[t])  # probs for next char
        return xs, hs, ycap

    def backward(self, xs, hs, ps, targets):
        # backward pass: compute gradients going backwards
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        dhnext = np.zeros_like(hs[0])
        for t, target in reversed(list(enumerate(targets))):
            dy = np.copy(ps[t])
            # through softmax
            dy[targets[t]] -= 1  # backprop into y
            # calculate dV, dc
            dV += np.outer(dy, hs[t].T)
            dc += dy
            # dh includes gradient from two sides, next cell and current output
            dh = np.dot(self.V.T, dy) + dhnext  # backprop into h
            # backprop through tanh non-linearity
            dhrec = (1 - hs[t] * hs[t]) * dh  # dhrec is the term used in many equations
            db += dhrec
            # calculate dU and dW
            dU += np.outer(dhrec, xs[t].T)
            dW += np.outer(dhrec, hs[t - 1].T)
            # pass the gradient from next cell to the next iteration.
            dhnext = np.dot(self.W.T, dhrec)
        # clip to mitigate exploding gradients
        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, 5, out=dparam)
        return dU, dW, dV, db, dc

    def loss(self, ps, targets):
        """loss for a sequence"""
        # calculate cross-entrpy loss
        return sum(-np.log(ps[t][target]) for t, target in enumerate(targets))

    def update_model(self, dU, dW, dV, db, dc, eps=1e-10):
        # parameter update with adagrad
        for param, dparam, momentum in zip([self.U, self.W, self.V, self.b, self.c],
                                      [dU, dW, dV, db, dc],
                                      [self.mU, self.mW, self.mV, self.mb, self.mc]):
            momentum += dparam ** 2
            param -= self.eta * dparam / np.sqrt(momentum + eps)  # adagrad update

    def synthesize(self, n, x0='.', h=None):
        if h is None:
            h = np.zeros(self.hidden_size)
        chars = list(x0)
        assert len(chars) > 0, 'Input sequence must be longer than 0'
        output = []
        x = self.dataloader.char_to_one_hot(chars[0])
        for char in chars:
            x = self.dataloader.char_to_one_hot(char)
            h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
            output.append(char)

        for t in range(n):
            h = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)
            y = np.dot(self.V, h) + self.c
            p = softmax(y)
            idx = np.random.choice(self.input_size, p=p.ravel())
            char = self.dataloader.ind_to_char[idx]
            x = self.dataloader.char_to_one_hot(char)
            output.append(char)
        return ''.join(output)

    def train(self):
        for iteration in range(self.n_iterations):
            inputs, targets = self.dataloader.next_batch()
            xs, hs, ps = self.forward(inputs, self.hprev)
            dU, dW, dV, db, dc = self.backward(xs, hs, ps, targets)
            loss = self.loss(ps, targets)
            self.update_model(dU, dW, dV, db, dc)
            self.smooth_loss.append(self.smooth_loss[-1] * 0.999 + loss * 0.001)
            self.hprev = hs[self.seq_length - 1]
            if not iteration % 500 and iteration % 10000:
                print(f'Iteration {iteration}, Loss: {self.smooth_loss[-1]}')
            if not iteration % 10000:
                synth_text = self.synthesize(n=200, h=self.hprev, x0=self.dataloader.ind_to_char[inputs[0]])
                print(f'Iteration {iteration}, Loss: {self.smooth_loss[-1]}, Synthesized text: {synth_text}')
                save_text('generated_text', f'Iteration {iteration}: Synthesized text: {synth_text}\n')
            if not iteration % 100000:
                synth_text = self.synthesize(n=1000, h=self.hprev, x0=self.dataloader.ind_to_char[inputs[0]])
                print(f'Iteration {iteration}, Loss: {self.smooth_loss[-1]}, Synthesized text: {synth_text}')
                save_text('generated_text', f'Iteration {iteration}: Synthesized text: {synth_text}\n')

    def save_model(self, fn):
        with open(f"{fn}.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f'Saved model to {fn}.pkl')

    def plot_loss(self):
        plt.plot(self.smooth_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Smooth loss')
        plt.show()

    def compute_gradients_num(self, inputs, targets, h):

        grads = []

        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)

        hprev = np.zeros(self.hidden_size)
        xs, hs, ps = self.forward(inputs, hprev)
        c = self.loss(ps, targets)

        for i in tqdm(range(self.b.shape[0])):
            self.b[i] += h
            _, _, ps = self.forward(inputs, hprev)
            c2 = self.loss(ps, targets)
            db[i] = (c2 - c) / h
            self.b[i] -= h

        for i in tqdm(range(self.c.shape[0])):
            self.c[i] += h
            _, _, ps = self.forward(inputs, hprev)
            c2 = self.loss(ps, targets)
            dc[i] = (c2 - c) / h
            self.c[i] -= h

        for i in tqdm(range(self.U.shape[0])):
            for j in range(self.U.shape[1]):
                self.U[i, j] += h
                _, _, ps = self.forward(inputs, hprev)
                c2 = self.loss(ps, targets)
                dU[i, j] = (c2-c) / h
                self.U[i, j] -= h

        for i in tqdm(range(self.V.shape[0])):
            for j in range(self.V.shape[1]):
                self.V[i, j] += h
                _, _, ps = self.forward(inputs, hprev)
                c2 = self.loss(ps, targets)
                dV[i, j] = (c2-c) / h
                self.V[i, j] -= h

        for i in tqdm(range(self.W.shape[0])):
            for j in range(self.W.shape[1]):
                self.W[i, j] += h
                _, _, ps = self.forward(inputs, hprev)
                c2 = self.loss(ps, targets)
                dW[i, j] = (c2-c) / h
                self.W[i, j] -= h

        grads.append(dU)
        grads.append(dW)
        grads.append(dV)
        grads.append(db)
        grads.append(dc)

        return grads



