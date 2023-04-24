import matplotlib.pyplot as plt

from classification_network import ClassificationNetwork
import numpy as np


def main():
    class_net = ClassificationNetwork(reg_factor=0.001, n_epochs=100, n_batch=100,
                                      hidden_size=400,
                                      n_train_samples=49000, n_val_samples=1000,
                                      cyclical_lr=[1e-5, 1e-1, 1000, 8, 1],
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


