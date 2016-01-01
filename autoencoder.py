import numpy as np
import tensorflow as tf


class TFAutoEncoder(object):
    """class for Auto Encoder on Tensorflow"""

    def __init__(self, target_dim, hidden_units=[], learning_rate=0.1,
                        tf_master="", num_cores=4, steps=50, logdir=None):
        self.hidden_units = hidden_units
        self.hidden_num = len(hidden_units)
        self.target_dim = target_dim
        self.learning_rate = learning_rate
        self.steps = steps
        self.tf_master = tf_master
        self.num_cores = num_cores
        self.logdir = logdir


    def fit(self, data):
        data = np.array(data)
        shape = data.shape
        if len(shape) != 2:
            raise TypeError("The shape of data is invalid")
        self.input_dim = shape[1]
        self._setup_graph()
        sess = self._session
        if self.logdir:
            writer = tf.train.SummaryWriter(self.logdir, sess.graph_def)
        sess.run(self._initializer)
        for step in range(self.steps):
            feed_dict = {self._input: data,
                                self._batch_size: float(shape[0])}
            loss, summ, _ = sess.run([self._loss, self._summ, self._optimize],
                                                    feed_dict=feed_dict)
            if self.logdir:
                writer.add_summary(summ, step)

    def encode(self, data):
        sess = self._session
        encoded = sess.run(self._encoded,
                                    feed_dict={self._input: data})
        return encoded

    def reconstructe(self, data):
        sess = self._session
        reconstructed = sess.run(self._reconstructed,
                                            feed_dict={self._input: data})
        return reconstructed

    def _setup_graph(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            input_dim, target_dim = self.input_dim, self.target_dim
            hidden_units = self.hidden_units
            lr = self.learning_rate

            self._input = X = tf.placeholder(name="X", dtype="float",
                                                    shape=[None, self.input_dim])
            self._batch_size = batch_size = tf.placeholder(name="batchsize", dtype="float")

            layer_units = [input_dim] + hidden_units + [target_dim]
            layer_num = len(layer_units)

            self._W_list = W_list = []
            self._b_list = b_list = []

            #weight and bias variables for encode part
            for n in range(layer_num-1):
                w_shape = layer_units[n: n+2]
                W = self._weight_variable(w_shape)
                W_list.append(W)
                b_shape = [layer_units[n+1]]
                b = self._bias_variable(b_shape)
                b_list.append(b)
            #bias variables for decode part
            for n in reversed(range(layer_num-1)):
                b_shape = [layer_units[n]]
                b = self._bias_variable(b_shape)
                b_list.append(b)

            #difinition of encode part
            outputs = [X]
            for n in range(layer_num-1):
                n_input = outputs[n]
                W_n = W_list[n]
                b_n = b_list[n]
                n_output = tf.nn.relu(tf.matmul(n_input, W_n) + b_n)
                outputs.append(n_output)

            #encoded input
            self._encoded = outputs[-1]

            #difinition of decode parts
            for n in range(layer_num - 1, 2*layer_num - 2):
                n_input = outputs[n]
                W_n = tf.transpose(W_list[2*layer_num - 3 - n])
                b_n = b_list[n]
                n_output = tf.nn.relu(tf.matmul(n_input, W_n) + b_n)
                outputs.append(n_output)

            #reconstructed input
            self._reconstructed = X_ = outputs[-1]

            self._loss = loss = tf.nn.l2_loss(X - X_) / batch_size
            self._optimize = tf.train.GradientDescentOptimizer(lr).minimize(loss)

            #variables summary
            tf.scalar_summary("l2_loss", self._loss)
            self._summ = tf.merge_all_summaries()

            #create session
            self._session = tf.Session(self.tf_master,
                                                config=tf.ConfigProto(
                                                    inter_op_parallelism_threads=self.num_cores,
                                                    intra_op_parallelism_threads=self.num_cores))
            #create initializer
            self._initializer = tf.initialize_all_variables()


    def _weight_variable(self, shape, stddev=0.1):
        init = tf.random_normal(shape, stddev)
        w = tf.Variable(init)
        return w

    def _bias_variable(self, shape):
        init = tf.ones(shape=shape)
        return tf.Variable(init)

if __name__ == '__main__':
    ae = TFAutoEncoder(target_dim=3, hidden_units=[5], num_cores=8,
                                    logdir="testlog")
    X = np.arange(50000).reshape(5000, 10)
    ae.fit(X)
    Y = ae.reconstructe(X)
    print(Y)
