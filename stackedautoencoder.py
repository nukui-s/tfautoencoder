"""Stacked Autoencoder on Tensorflow"""

#Authors: Nukui Shun <nukui.s@ai.cs.titech.ac.jp>
#License : GNU General Public License v2.0

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from six.moves import xrange

import numpy as np
import tensorflow as tf

from autoencoder import TFAutoEncoder
from autoencoder import ACTIVATE_FUNC

class TFStackedAutoEncoder(object):
    """Class for Stacked Auto Encoder on Tensorflow

        Attributes (layer_units)
        --------------------
        layer_units : The number of units in each layer.

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
    """

    def __init__(self, layer_units, learning_rate=0.01, noising=False,
                        noise_stddev=10e-2, activate_func="relu", optimizer="adam",
                        steps=50, w_stddev=0.1, lambda_w=1.0, num_cores=4,
                        logdir=None):
        self.layer_units = layer_units
        self.layer_num = len(layer_units)
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_cores = num_cores
        self.w_stddev = w_stddev
        self.lambda_w = lambda_w
        self.noising = noising
        self.noise_stddev = noise_stddev
        self.activate_func = activate_func
        self.optimizer = optimizer
        self.logdir = logdir

    def fit(self, data):
        """Optimize stacked autoencoder from the bottom layer"""
        data = np.array(data)
        shape = data.shape
        if len(shape) != 2:
            raise TypeError("The shape of data is invalid")
        if shape[1] != self.layer_units[0]:
            raise ValueError("Input dimension and 1st layer units does not match")
        outputs = [data]
        weights = []
        biases = []
        for n in xrange(1, self.layer_num):
            input_n = outputs[n-1]
            hidden_dim = self.layer_units[n]
            output_n, W_n, b_n = self._partial_fit(input_n, hidden_dim, n)
            outputs.append(output_n)
            weights.append(W_n)
            biases.append(b_n)
        self.weights = weights
        self.biases = biases
        self._setup_encode()

    def encode(self, data, layer=-1):
        """Encode data by learned stacked autoencoder """
        sess = self.session
        encoded = sess.run(self._outputs[layer],
                                    feed_dict={self._input: data})
        return encoded

    def _setup_encode(self):
        """Setup computation graph for encode """
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._input = X = tf.placeholder("float", shape=[None, self.layer_units[0]])

            #setup data nodes for weights and biases
            weights = []
            biases = []
            for n, w in enumerate(self.weights):
                name = "weight_" + str(n)
                weights.append(tf.Variable(w, name=name))
            for n, b in enumerate(self.biases):
                name = "bias_" + str(n)
                biases.append(tf.Variable(b, name=name))

            #define encoder: outputs[-1] is final result of encoding
            self._outputs = outputs = [X]
            activate_func = ACTIVATE_FUNC[self.activate_func]
            for n in xrange(self.layer_num-1):
                x = outputs[n]
                w = weights[n]
                b = biases[n]
                output_n = activate_func(tf.matmul(x, w) + b)
                outputs.append(output_n)

            #create session
            self.session = tf.Session(config=tf.ConfigProto(
                                                    inter_op_parallelism_threads=self.num_cores,
                                                    intra_op_parallelism_threads=self.num_cores))
            init_op = tf.initialize_all_variables()
            self.session.run(init_op)

    def _partial_fit(self, data, hidden_dim, layer_no):
        """Optimize single autoencoder"""
        #make logdir for each layer
        if self.logdir:
            logdir = self.logdir + "/layer" + str(layer_no)
            if not os.path.exists(logdir):
                os.makedirs(logdir)
        else:
            logdir = self.logdir
        ae = TFAutoEncoder(hidden_dim=hidden_dim,
                                        learning_rate=self.learning_rate,
                                        noising=self.noising,
                                        noise_stddev=self.noise_stddev,
                                        w_stddev=self.w_stddev,
                                        lambda_w=self.lambda_w,
                                        steps=self.steps,
                                        activate_func=self.activate_func,
                                        optimizer=self.optimizer,
                                        continue_training=False,
                                        logdir=logdir,
                                        num_cores=self.num_cores)
        ae.fit(data)
        output = ae.encode(data)
        weight = ae.weight
        bias = ae.bias[0]
        return output, weight, bias
