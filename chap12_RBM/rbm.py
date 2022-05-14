# python: 2.7
# encoding: utf-8

import numpy as np


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""
        self.n_visible = n_observe
        self.n_hidden = n_hidden
        self.bias_a = np.zeros(self.n_visible)  # 可视层偏移量
        self.bias_b = np.zeros(self.n_hidden)  # 隐藏层偏移量
        self.weights = np.random.normal(0, 0.01, size=(self.n_visible, self.n_hidden))
        self.n_sample = None

    def train(self, data):
        """Train model using data."""

        self.n_sample = data.shape[0]

        v_cd = self.gibbs_sample(data, max_cd)
        self.update(data, v_cd, eta)
        error = np.sum((data - v_cd) ** 2) / self.n_sample / self.n_visible * 100
        if not i % 100:  # 将重构后的样本与原始样本对比计算误差
            print("可视层状态误差比例:{0}%".format(round(error, 2)))


    def sample(self):
        """Sample from trained model."""
        v = v0
        # 首先根据输入样本对每个隐藏层神经元采样。二项分布采样，决定神经元是否激活
        ph = self.encode(v)
        h = np.random.binomial(1, ph, (self.n_sample, self.n_hidden))
        # 根据采样后隐藏层神经元取值对每个可视层神经元采样
        pv = self.decode(h)
        v = np.random.binomial(1, pv, (self.n_sample, self.n_visible))


# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print(mnist.shape)

    # construct rbm model
    rbm = RBM(2, img_size)

    # train rbm model using mnist
    rbm.train(mnist)

    # sample from rbm model
    s = rbm.sample()
