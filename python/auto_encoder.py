#coding:utf-8
import os
import sys
import time
import timeit
import gzip
import cPickle
import numpy
import numpy.random as RandomState
from numpy import *

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams # theano中随机数生成器
from theano.tensor.sharedvar import TensorSharedVariable
from theano.compile.sharedvalue import shared

# import settings

"""
其中参数input, W, bhid, bvis都是张量类型：theano.tensor.TensorType
设定随机数: numpy.random.RandomState; theano.tensor.shared_randomstreams.RandomStreams
如果最初没给权重W，则初始化为[-a, a]之间的均匀分布，numpy.random.RandomState.uniform(low, high, size)
"""
def load_data(dataset):

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        self.activation = activation
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def get_output(self, input):
        lin_output = T.dot(input, self.W) + self.b
        output = (
                lin_output if self.activation is None
                else self.activation(lin_output)
                )
        return output

class AutoEncoder(object):

    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=100,
            n_hidden=20, W=None, bhid=None, bvis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

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
        self.theano_rng = theano_rng

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

class NaiveStackedAutoEncoder(object):
    """
    我称之为朴素栈式自编码器。
    此编码器专门针对单个case，对此case进行栈式自编码。主要目的是降维。
    """

    def __init__(self,set_x, n_layer_sizes):

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

            for epoch in range(settings.MAX_EPOCH):
                cost_value, hidden_value, construction_value = train_autoencoder(layer_input)

            endTime = time.time()
            print ">>>layer:%d, time cost:%f" % (i, (endTime - startTime))
            # print ">>>W is ", autoencoder.W.get_value()
            # print ">>>hidden is ", hidden_value
            # print ">>>output is ", autoencoder.get_reconstructed_input(hidden_value).eval()
            self.hiddens.append(hidden_value)

def get_features_using_autoencoder(set_x, layer_sizes):
    """
    deprecated
    """
    N = len(set_x)
    set_x = [set_x]
    layer_sizes.insert(0, N)
    sda = NaiveStackedAutoEncoder(set_x, layer_sizes)
    hiddens = sda.hiddens
    features = hiddens[-1]
    return features

class AdvancedStackedAutoEncoder(object):
    """
    我称之为加强版栈式自编码
    此栈式自编码器接收多个train_set进行训练。然后利用训练后的参数
    对test_set进行降维
    """

    def __init__(self, numpy_rng, n_ins=784,
            hidden_layers_sizes=[500, 500]):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = AutoEncoder(numpy_rng=numpy_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2

    def pretraining_functions(self, train_set_x, batch_size):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def get_features(self, test_case):
        """
        generate features once training completed
        """
        hidden_values = []
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = test_case
            else:
                layer_input = hidden_values[-1]

            hidden_value = self.sigmoid_layers[i].get_output(layer_input)
            hidden_values.append(hidden_value)
            
        # return the final hidden values
        return hidden_values[-1]

class OriginalStackedAutoEncoder(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=50,
    ):
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        print "layers sizes", hidden_layers_sizes

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            dA_layer = AutoEncoder(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        self.errors = self.logLayer.errors(self.y)


    def pretraining_functions(self, train_set_x, batch_size):
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        # n_valid_batches = 10000
        # n_test_batches = 10000


        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def get_features(self, test_case):
        """
        generate features once training completed
        """
        hidden_values = []
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = test_case
            else:
                layer_input = hidden_values[-1]

            hidden_value = self.sigmoid_layers[i].get_output(layer_input)
            hidden_values.append(hidden_value)
            
        # return the final hidden values
        return hidden_values[-1]

class StackedAutoEncoderDriver(object):
    """
    此类实现对外API, 使用时只需要先调用train函数进行训练，
    然后再调用get_features传入test_set获取最后的features即可
   """

    def __init__(self):
        self.sda = None

    def train_with_mnist(self, finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1, hidden_layers_sizes=[500, 500, 50]):
        datasets = load_data(dataset)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        # n_train_batches = 10000

        # start-snippet-3
        numpy_rng = numpy.random.RandomState(89677)
        print '... building the model'
        # construct the stacked denoising autoencoder class
        self.sda = OriginalStackedAutoEncoder(
            numpy_rng=numpy_rng,
            n_ins=28 * 28,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=10
        )
        # end-snippet-3 start-snippet-4
        print '... getting the pretraining functions'
        pretraining_fns = self.sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size)

        print '... pre-training the model'
        start_time = timeit.default_timer()
        for i in xrange(self.sda.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                epoch_start = time.time()
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                             lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c),
                print ', time cost:', (time.time() - epoch_start)

        end_time = timeit.default_timer()

        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

        # end-snippet-4
        ########################
        # FINETUNING THE MODEL #
        ########################

        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = self.sda.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )

        print '... finetunning the model'
        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0

        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = test_model()
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def train(self, train_set_x, pretraining_epochs=15,
            pretrain_lr=0.001, batch_size=1, n_ins=784,
            hidden_layers_sizes=[500, 500]):
        """
        对StackedAutoEncoder进行训练
        """

        if not isinstance(train_set_x, TensorSharedVariable):
            train_set_x = shared(train_set_x)
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size

        print "hidden_layers_sizes: ", hidden_layers_sizes

        print "... building the model"
        numpy_rng = numpy.random.RandomState(89677)
        self.sda = AdvancedStackedAutoEncoder(
            numpy_rng=numpy_rng,
            n_ins=n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
        )

        print "... getting the pretraining function"
        pretraining_fns = self.sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
        print '... pre-training the model'
        for i in xrange(self.sda.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)

    def get_features(self, test_case):
        """
        特征提取
        """
        features = self.sda.get_features(test_case)
        return features.eval()


def test_stacked_autoencoder():
    """
    test NaiveStackedAutoEncoder
    """
    N = 5
    #set_x = numpy.random.randn(1, 100)
    set_x = [1 if i%3== 0 or i%7==0 else 0 for i in range(N)]
    set_x = [set_x]
    print ">>> set_x", set_x
    sda = NaiveStackedAutoEncoder(set_x, n_layer_sizes=[N, 2, 1])

def test_SdA():
    """
    test AdvancedStackedAutoEncoder
    """
    # test Sda 就是这么搞
    # 现在想要得到feature只需要执行两个接口即可
    # train(), get_features() 非常easy
    train_sets = [
            [1., 1., 1.],
            [2., 2., 2.],
        ]

    train_set_x = numpy.asarray(train_sets)
    train_set_x = shared(train_set_x)
    # test_set = [4.] * (28*28)
    test_set = [4.] * 3
    driver = StackedAutoEncoderDriver()
    driver.train(train_set_x, n_ins=3, hidden_layers_sizes=[2, 1])
    # driver.train_with_mnist()

    params = driver.sda.params
    features = driver.get_features(test_set)
    print features

if __name__ == "__main__":
    # test_autoencoder()
    # test_stacked_autoencoder()
    test_SdA()
