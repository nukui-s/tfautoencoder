from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tfautoencoder import TFAutoEncoder
from tfautoencoder import TFStackedAutoEncoder


class Test(tf.test.TestCase):
    def testAutoEncoder(self):
        encoder = TFAutoEncoder(hidden_dim=3)
        X = np.random.rand(50000).reshape(5000,10)
        encoder.fit(X)
        Y = encoder.encode(X)
        X_ = encoder.reconstruct(X)

    def testStackedAE(self):
        encoder = TFStackedAutoEncoder(layer_units=[100, 30, 10])
        X = np.arange(100000).reshape(1000,100)
        encoder.fit(X)
        Y = encoder.encode(X)
        weight = encoder.weight
        bias = encoder.bias

if __name__ == '__main__':
    tf.test.main()
