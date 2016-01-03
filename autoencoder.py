"""Autoencoder on Tensorflow"""

# Authors: Nukui Shun <nukui.s@ai.cs.titech.ac.jp>
#Licencse: BSD 3 clause

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from six.moves import xrange

import numpy as np
import tensorflow as tf

ACTIVATE_FUNC = {"relu": tf.nn.relu,
                                "sigmoid": tf.nn.sigmoid,
                                "softplus": tf.nn.softplus}

OPTIMIZER = {"sgd": tf.train.GradientDescentOptimizer,
                        "adagrad": tf.train.AdagradOptimizer,
                        "adam": tf.train.AdamOptimizer}

class TFAutoEncoder(object):
    """class for Auto Encoder on Tensorflow

        Attributes (hidden_dim)
        --------------------
        hidden_dim : The number of units in hidden layer

        learning_rate : Learning rate used in optimization

        noising : If True fitting as denoising autoencoder(defalut is False)

        noise_stddev : Only used when noising is True

        activate_func : Selected in 'relu'(default) 'softplus' 'sigmoid'

        optimizer : Selected in 'sgd' 'adagrad' 'adam'(default)

        steps : The number of steps in optimization

        w_stddev : Used in initializing weight variables

        lambda_w : Penalty coefficient of regularization term

        num_cores : The number of cores used in Tensorflow computation

        logdir : Directory to export log

        continue_training : If True keep the learned state at each time
                                        performing fit()
    """

    def __init__(self, hidden_dim, learning_rate=0.01, noising=False,
                        noise_stddev=10e-2, activate_func="relu", optimizer="adam",
                        steps=50, w_stddev=0.1, lambda_w=1.0, num_cores=4,
                        logdir=None, continue_training=False):
        if not activate_func in ACTIVATE_FUNC:
            raise ValueError("activate_func must be chosen of the following:"
                                    "`relu``sigmoid` `softplus`")
        if not optimizer in OPTIMIZER:
            raise ValueError("optimizer must be chosen of the following:"
                                    "`sgd` `adagrad` `adam`")
        self.activate_func = activate_func
        self.optimizer = optimizer
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.noising = noising
        self.steps = steps
        self.num_cores = num_cores
        self.w_stddev = w_stddev
        self.noise_stddev = noise_stddev
        self.lambda_w = lambda_w
        self.continue_training = continue_training
        self.logdir = logdir
        self._initialized = False

    def fit(self, data):
        """Optimize autoencoder"""
        data = np.array(data)
        shape = data.shape
        if len(shape) != 2:
            raise TypeError("The shape of data is invalid")
        self.input_dim = shape[1]

        #setup computational graph if not initialized
        if not (self.continue_training and self._initialized):
            self._setup_graph()
        sess = self._session

        #setup summary writer for TensorBoard
        if self.logdir:
            writer = tf.train.SummaryWriter(self.logdir, sess.graph_def)

        for step in xrange(self.steps):
            feed_dict = self._get_feed_dict(data, self.noising)
            loss, summ, _ = sess.run([self._loss, self._summ, self._optimize],
                                                    feed_dict=feed_dict)
            if self.logdir:
                writer.add_summary(summ, step)
        self.fit_loss = loss


    def encode(self, data):
        """Encode data by learned autoencoder """
        sess = self._session
        feed_dict = self._get_feed_dict(data, noising=False)
        encoded = sess.run(self._encoded,
                                    feed_dict=feed_dict)
        return encoded

    def reconstruct(self, data):
        """Encode and decode input data"""
        sess = self._session
        feed_dict = self._get_feed_dict(data, noising=False)
        reconstructed = sess.run(self._reconstructed,
                                            feed_dict=feed_dict)
        return reconstructed

    def _get_feed_dict(self, data, noising):
        shape = data.shape
        feed_dict = {self._input: data,
                        self._batch_size: float(shape[0])}
        if noising:
            noise = self._generate_noise(shape, self.noise_stddev)
            feed_dict[self._noise] = noise
        else:
            zeros = np.zeros(shape=shape)
            feed_dict[self._noise] = zeros
        return feed_dict

    def _setup_graph(self):
        """Setup computation graph for training"""
        self._graph = tf.Graph()
        with self._graph.as_default():
            input_dim, hidden_dim = self.input_dim, self.hidden_dim
            lr = self.learning_rate
            activate_func = ACTIVATE_FUNC[self.activate_func]
            optimizer = OPTIMIZER[self.optimizer]

            self._input = X = tf.placeholder(name="X", dtype="float",
                                                        shape=[None, input_dim])
            self._batch_size = batch_size = tf.placeholder(name="batchsize",
                                                                                    dtype="float")
            self._noise = noise = tf.placeholder(name="noise", dtype="float",
                                                                shape=[None, input_dim])
            clean_X = X
            X = X + noise

            self._W = W = self._weight_variable(shape=[input_dim, hidden_dim],
                                                                    stddev=self.w_stddev)

            self._b = b = self._bias_variable([hidden_dim])
            self._c = c = self._bias_variable([input_dim])

            encoded = activate_func(tf.matmul(X, W) + b)
            self._encoded = encoded

            Wt = tf.transpose(W)
            reconstructed = activate_func(tf.matmul(encoded, Wt) + c)
            self._reconstructed = reconstructed

            regularizer = self.lambda_w * (tf.nn.l2_loss(W) + \
                                                        tf.nn.l2_loss(b) + \
                                                        tf.nn.l2_loss(c))
            error = tf.nn.l2_loss(clean_X - reconstructed)
            self._loss = loss = error + regularizer
            self._optimize = optimizer(lr).minimize(loss)

            #variables summary
            tf.scalar_summary("l2_loss", loss)
            self._summ = tf.merge_all_summaries()

            #create session
            self._session = tf.Session(config=tf.ConfigProto(
                                                    inter_op_parallelism_threads=self.num_cores,
                                                    intra_op_parallelism_threads=self.num_cores))
            #create initializer
            self._initializer = tf.initialize_all_variables()
            self._session.run(self._initializer)
            self._initialized = True

    @classmethod
    def _weight_variable(cls, shape, stddev):
        """Generate weight matrix variable by normal distribution"""
        init = tf.truncated_normal(shape, stddev)
        w = tf.Variable(init)
        return w

    @classmethod
    def _bias_variable(cls, shape):
        """Generate bias vector variable by ones"""
        #init = tf.truncated_normal(shape, stddev)
        init = tf.ones(shape)
        b = tf.Variable(init)
        return b

    @classmethod
    def _generate_noise(cls, shape, stddev):
        """Generate noise for denoising autoencoder"""
        noise = np.random.normal(size=shape, scale=stddev)
        return noise

    @property
    def weight(self):
        """Return weight as numpy array"""
        sess = self._session
        weight = self._W
        return sess.run(weight)

    @property
    def bias(self):
        """Return bias as numpy array"""
        sess = self._session
        b, c = self._b, self._c
        b = sess.run(b)
        c = sess.run(c)
        return b, c
