import matplotlib.pyplot as plt

from classification_network import ClassificationNetwork
import numpy as np


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


