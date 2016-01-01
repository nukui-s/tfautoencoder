from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from autoencoder import TFAutoEncoder

class Test(tf.test.TestCase):
    def testAutoEncoder(self):
        with self.test_session() as sess:
            encoder = TFAutoEncoder(target_dim=3, hidden_units=[5])
            X = np.arange(50).reshape(5,10)
            encoder.fit(X)
            Y = encoder.encode(X)
            print(X)
            print(Y)
            #tf.initialize_all_variables().run()
            #feature = encoder.reduced_feature
            #print(sess.run(feature, feed_dict={encoder.X: X}))


if __name__ == '__main__':
    tf.test.main()
