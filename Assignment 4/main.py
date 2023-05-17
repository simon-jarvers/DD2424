import numpy as np
import matplotlib.pyplot as plt

from recurrent_network import RNN
from utils import load_model


def main():
    book_fname = 'goblet_book.txt'
    rnn: RNN = RNN(book_fname, n_iterations=100001)

    # check_gradients(rnn)

    # rnn: RNN = load_model('model')
    rnn.train()
    rnn.plot_loss()
    rnn.save_model('model')


def check_gradients(rnn):
    hprev = np.zeros(rnn.hidden_size)
    inputs, targets = rnn.dataloader.next_batch()
    xs, hs, ps = rnn.forward(inputs, hprev)

    grads_num = rnn.compute_gradients_num(inputs, targets, h=1e-5)
    grads_ana = list(rnn.backward(xs, hs, ps, targets))
    names = ['U', 'V', 'W', 'b', 'c']
    for i, (grad_ana, grad_num) in enumerate(zip(grads_ana, grads_num)):
        rel_diffs = []
        abs_diffs = []
        for a, n in zip(grad_ana.ravel(), grad_num.ravel()):
            rel_diff = np.abs(a - n)/max(np.float64(1e-1), (np.abs(a) + np.abs(n)))
            rel_diffs.append(rel_diff)
            abs_diff = np.abs(a - n)
            abs_diffs.append(abs_diff)

        fig, ax = plt.subplots(1, 2, figsize=(7, 12))
        ax[0].boxplot(rel_diffs)
        ax[0].set_title(f'Relative differences for {names[i]}')
        ax[1].boxplot(abs_diffs)
        ax[1].set_title(f'Absolute differences for {names[i]}')
        plt.show()


if __name__ == '__main__':
    main()

