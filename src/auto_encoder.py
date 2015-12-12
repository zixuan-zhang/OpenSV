
#coding:utf-8
import time
import numpy
import numpy.random as RandomState
from numpy import *

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams # theano中随机数生成器

"""
其中参数input, W, bhid, bvis都是张量类型：theano.tensor.TensorType
设定随机数: numpy.random.RandomState; theano.tensor.shared_randomstreams.RandomStreams
如果最初没给权重W，则初始化为[-a, a]之间的均匀分布，numpy.random.RandomState.uniform(low, high, size)
"""

class AutoEncoder(object):

    def __init__(self, numpy_rng, input=None, n_visible=100,
            n_hidden=20, W=None, bhid=None, bvis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                low= -4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                size = (n_visible, n_hidden)),
                dtype='float64')
            W = theano.shared(value=initial_W, name="W") # 设置为共享变量

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype="float64"), name="bvis")
        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype="float64"), name="bhid")

        self.W = W
        self.b = bhid # b 对应hidden的bias
        self.b_prime = bvis # b_prime对应 input 层的bias
        self.W_prime = self.W.T # 转置

        if not input:
            self.x = T.dmatrix(name="input") # TensorType variable: 2-dim matrix type
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b) # 返回隐藏层值

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    # 将所有数据点(data point样本），得到的损失函数是一个向量，axis=1表示按列计算
    # 将上式得到的所有样本的重构误差之和，下面计算平均重构误差，实际的损失函数
    # 计算目标函数的梯度
    def get_cost_updates(self, learning_rate):

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)

        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params) # 张量类型数据，可以方便的球梯度T.grad

        # 生成参数的更新列表，实际中可能没必要保存所有的更新参数，只需要最后一组即可
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param-learning_rate*gparam))

        return cost, updates

def test_autoencoder():
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    M = 100 # 数据的case数量
    N = 20 # 一个case的维数
    K = 10 # 降维后的维数 
    train_x = numpy.random.randn(M, N) # 用随机数生成数据

    # 构建类型，初始化自编码器
    # numpy_rng 用来初始化参数W
    autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(1234), 
            input=x, n_visible=N, n_hidden=K) 

    cost, updates = autoencoder.get_cost_updates(learning_rate=0.1)
    hidden = autoencoder.get_hidden_values(x)
    construction = autoencoder.get_reconstructed_input(hidden)

    # 训练模型
    startTime = time.time()
    train_autoencoder = theano.function(inputs = [x],
            outputs = [cost, hidden, construction], updates = updates,
            on_unused_input="ignore")

    # 在训练时才把训练数据传给训练函数
    # 我的理解是，在训练函数的定义中，inputs=[x]，这个x是一个共享变量
    # 然而这个共享变量x在初始化autoencoder的时候赋给了self.x。因此x与
    # self.x建立连接，也就是同一个共享变量。
    # 在训练时，传给训练函数的参数是train_x，也就是实际的数据，此时，
    # train_x与x建立连接，也就是将train_x的数据传给了x，x里的数据就是
    # train_x。
    cost_value, hidden_value, construction_value = train_autoencoder(train_x)
    endTime = time.time()

    print "shape(autoencoder.W): %s, \ttype(autoencoder.W):%s" % (
            shape(autoencoder.W.get_value()), type(autoencoder.W.get_value()))
    print "autoencoder.W.get_value():\n", autoencoder.W.get_value()

    print "type(train_autoencoder):", type(train_autoencoder)
    print "cost_value:", cost_value

    print "shape(autoencoder.get_hidden_values(train_x)):", shape(hidden_value)
    print "shape(autoencoder.get_reconstructed_input(train_x))", shape(construction_value)

    # print hidden_value
    # print construction_value

class StackedAutoEncoder(object):
    """
    stacked autoencoder
    """

    def __init__(self,set_x, n_layer_sizes, ):

        self.hiddens = []
        
        for i in xrange(len(n_layer_sizes) - 1):
            if i == 0:
                layer_input = set_x
            else:
                layer_input = self.hiddens[-1]
            n_input_size = n_layer_sizes[i]
            n_output_size = n_layer_sizes[i+1]
            x = T.dmatrix('x' + str(i))

            # print "iterator: %d, n_input: %d, n_output: %d" % (i, n_input_size, n_output_size)

            autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(1234),
                    input=x, n_visible=n_input_size,
                    n_hidden=n_output_size)

            cost, updates = autoencoder.get_cost_updates(learning_rate=0.1)
            hidden = autoencoder.get_hidden_values(x)
            construction = autoencoder.get_reconstructed_input(hidden)

            startTime = time.time()
            train_autoencoder = theano.function(inputs=[x],
                    outputs=[cost, hidden, construction], updates=updates)

            for epoch in range(1000):
                cost_value, hidden_value, construction_value = train_autoencoder(layer_input)

            endTime = time.time()
            print ">>>layer:%d, time cost:%f" % (i, (endTime - startTime))
            self.hiddens.append(hidden_value)
            print ">>>hidden_values are: "
            print hidden_value

def get_features_using_autoencoder(set_x, layer_sizes):
    N = len(set_x)
    set_x = [set_x]
    layer_sizes.insert(0, N)
    sda = StackedAutoEncoder(set_x, layer_sizes)
    hiddens = sda.hiddens
    features = hiddens[-1]
    return features

def test_stacked_autoencoder():
    N = 100
    #set_x = numpy.random.randn(1, 100)
    set_x = [205 if i%3== 0 or i%7==0 else 0 for i in range(100)]
    set_x = [set_x]
    sda = StackedAutoEncoder(set_x, n_layer_sizes=[100, 50, 20, 10])

if __name__ == "__main__":
    test_autoencoder()
