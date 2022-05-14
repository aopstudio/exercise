# python: 3.5.2
# encoding: utf-8

import numpy as np
import tensorflow as tf

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
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self):
        # 请补全此处代码
        self.W = tf.Variable(shape=[2, 1], dtype=tf.float32,
                             initial_value=tf.random.uniform(shape=[2, 1], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[1], dtype=tf.float32, initial_value=tf.zeros(shape=[1]))

        self.trainable_variables = [self.W, self.b]
        
        

    def train(self, data_train):
        """
        训练模型。
        """

        # 请补全此处代码
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        animation_fram = []
        for i in range(100):
            with tf.GradientTape() as tape:
                pred= self.predict(data_train[:,:2])
                losses=  tf.maximum(0,1-tf.constant(data_train[:, 2],dtype=tf.float32)*pred)
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
        pred = tf.squeeze(pred, axis=1)
        return pred
        # 请补全此处代码


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
