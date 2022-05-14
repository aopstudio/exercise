# python: 3.5.2
# encoding: utf-8


# 非线性问题

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

ndim=20

def gaussian_basis(x, feature_num=ndim):
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)

    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret

def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), dtype=tf.float32, depth=3)
    a=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, axis=1), tf.argmax(pred, axis=1)), dtype=tf.float32))
    return a

def func(data_train):
    basis_func = gaussian_basis
    x = data_train[:, 0]
    phi0 = np.expand_dims(np.ones_like(x), axis=1)
    phi1 = basis_func(x)
    x1 = np.concatenate([phi0, phi1], axis=1)

    x = data_train[:, 1]
    phi0 = np.expand_dims(np.ones_like(x), axis=1)
    phi1 = basis_func(x)
    x2 = np.concatenate([phi0, phi1], axis=1)

    xx = np.concatenate([x1, x2], axis=1)


    return xx


class SVM():
    """
    SVM模型。
    """

    def __init__(self):
        # 请补全此处代码

        self.W = tf.Variable(shape=[ndim*2+2, 3], dtype=tf.float32,
                             initial_value=tf.random.uniform(shape=[ndim*2+2, 3], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[1,3], dtype=tf.float32, initial_value=tf.zeros(shape=[1,3]))

        self.trainable_variables = [self.W, self.b]
        pass

    def train(self, data_train):
        """
        训练模型。
        """
        basis_func = gaussian_basis
        x = data_train[:, 0]
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        x1 = np.concatenate([phi0, phi1], axis=1)

        x = data_train[:, 1]
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        x2 = np.concatenate([phi0, phi1], axis=1)

        xx = np.concatenate([x1, x2], axis=1)


        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        animation_fram = []
        for i in range(1000):
            with tf.GradientTape() as tape:
                pred= self.predict(xx)
                label = tf.one_hot(tf.cast(tf.constant(data_train[:, 2],dtype=tf.float32), dtype=tf.int32), dtype=tf.float32, depth=3)
                losses=  tf.maximum(0,1-label*pred)
                loss= tf.reduce_mean(losses)

                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                animation_fram.append((self.W.numpy()[0, 0], self.W.numpy()[1, 0], self.b.numpy(), loss.numpy()))
                if i%10==0 :
                    print(loss)
        # 请补全此处代码
        return  animation_fram

    def predict(self, x):
        """
        预测标签。
        """
        pred = tf.matmul(tf.constant(x, dtype=tf.float32), self.W) + self.b

        pred = tf.nn.softmax(pred)

        # pred = tf.squeeze(pred, axis=1)
        # print(pred)



        return pred
        # 请补全此处代码


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    # train_file = 'data/train_linear.txt'
    # test_file = 'data/test_linear.txt'
    # train_file = 'data/train_kernel.txt'
    # test_file = 'data/test_kernel.txt'
    train_file = 'data/train_multi.txt'
    test_file = 'data/test_multi.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)



    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    animation_fram = svm.train(data_train)  # 训练模型

    # # 使用SVM模型预测标签
    x_train =func(data_train[:, :2])   # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    print(t_train_pred)
    x_test =func(data_test[:, :2])
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)
    #
    # # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))




    tensor_b1 = tf.greater(data_test[:, 2], 0)
    tensor_b= tf.boolean_mask(data_test, tensor_b1)

    tensor_c1 = tf.less(data_test[:, 2], 0)
    tensor_c = tf.boolean_mask(data_test, tensor_c1)

    tensor_d =tf.math.logical_not(tf.math.logical_or(tensor_b1 , tensor_c1))
    tensor_d = tf.boolean_mask(data_test, tensor_d)

    plt.figure()
    ax=plt.gca()
    plt.scatter(tensor_d[:, 0], tensor_d[:, 1], c='y', marker='.')
    plt.scatter(tensor_b[:, 0], tensor_b[:, 1], c='b', marker='+')
    plt.scatter(tensor_c[:, 0], tensor_c[:, 1], c='r', marker='o')
    line_d, = ax.plot([], [], label='fit_line')

    plt.show()
