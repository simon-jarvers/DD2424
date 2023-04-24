from classification_network import ClassificationNetwork
import numpy as np
from functions import ComputeGradsNum


def main():

    class_net = ClassificationNetwork(loss_function='bce', reg_factor=0.01, n_epochs=40, eta=.01, n_batch=100, after_n_epochs=10)

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


if __name__ == '__main__':
    main()


