from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from autoencoder import TFAutoEncoder

class Test(tf.test.TestCase):
    def testAutoEncoder(self):
        with self.test_session() as sess:
            encoder = TFAutoEncoder(target_dim=3, hidden_units=[5],
                                                     num_cores=8, steps=50)
            X = np.random.rand(50000).reshape(5000,10)
            encoder.fit(X)
            Y = encoder.encode(X)
            X_ = encoder.reconstructe(X)
            print(X)
            print(Y)
            print(X_)

if __name__ == '__main__':
    tf.test.main()
