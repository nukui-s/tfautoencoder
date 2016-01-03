import os
import numpy as np
import tensorflow as tf
from autoencoder import TFAutoEncoder

class TFStackedAutoEncoder(object):
    """class for Stacked Auto Encoder on Tensorflow"""

    def __init__(self, layer_units=[], learning_rate=0.1, noising=False,
                        noise_stddev=10e-2, activate_func="relu", optimizer="adam",
                        num_cores=4, steps=50, w_stddev=0.1, lambda_w=1.0,
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
        data = np.array(data)
        shape = data.shape
        if len(shape) != 2:
            raise TypeError("The shape of data is invalid")
        if shape[1] != self.layer_units[0]:
            raise ValueError("Input dimension and 1st layer units does not match")
        outputs = [data]
        weights = []
        biases = []
        for n in range(1, self.layer_num):
            print(n)
            input_n = outputs[n-1]
            hidden_dim = self.layer_units[n]
            output_n, W_n, b_n = self._partial_fit(input_n, hidden_dim, n)
            outputs.append(output_n)
            weights.append(W_n)
            biases.append(b_n)
        self.weights = weights
        self.biases = biases

    def _partial_fit(self, data, hidden_dim, layer_no):
        #make logdir for each layer
        if self.logdir:
            logdir = self.logdir + "/" + str(layer_no)
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

if __name__ == '__main__':
    sae = TFStackedAutoEncoder(layer_units=[10, 5, 5, 3], steps=10)
    X = np.arange(50000).reshape(5000, 10)
    sae.fit(X)
