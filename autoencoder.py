import numpy as np
import tensorflow as tf

ACTIVATE_FUNC = {"relu": tf.nn.relu,
                                "sigmoid": tf.nn.sigmoid,
                                "softplus": tf.nn.softplus}

class TFAutoEncoder(object):
    """class for Auto Encoder on Tensorflow"""

    def __init__(self, hidden_dim, learning_rate=0.1, noising=True,
                        noise_stddev=0.01, tf_master="", num_cores=4,
                        steps=50, w_stddev=0.1, activate_func="relu",
                        lambda_w=0, batch_size=100, logdir=None):
        if not activate_func in ACTIVATE_FUNC:
            raise ValueError("activate_func must be chosed of the following:"
                                    "`relu`,`sigmoid`, `softplus`")
        self.activate_func = activate_func
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.noising = noising
        self.steps = steps
        self.tf_master = tf_master
        self.num_cores = num_cores
        self.w_stddev = w_stddev
        self.noise_stddev = noise_stddev
        self.lambda_w = lambda_w
        self.batch_size = batch_size
        self.logdir = logdir


    def fit(self, data):
        data = np.array(data)
        shape = data.shape
        if len(shape) != 2:
            raise TypeError("The shape of data is invalid")
        self.input_dim = shape[1]

        #setup computational graph
        self._setup_graph()
        sess = self._session

        #setup summary writer for TensorBoard
        if self.logdir:
            writer = tf.train.SummaryWriter(self.logdir, sess.graph_def)

        sess.run(self._initializer)
        for step in range(self.steps):
            feed_dict = self._get_feed_dict(data, self.noising)
            loss, summ, _ = sess.run([self._loss, self._summ, self._optimize],
                                                    feed_dict=feed_dict)
            if self.logdir:
                writer.add_summary(summ, step)

    def encode(self, data):
        sess = self._session
        feed_dict = self._get_feed_dict(data, noising=False)
        encoded = sess.run(self._encoded,
                                    feed_dict=feed_dict)
        return encoded

    def reconstruct(self, data):
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
        self._graph = tf.Graph()
        with self._graph.as_default():
            input_dim, hidden_dim = self.input_dim, self.hidden_dim
            lr = self.learning_rate
            activate_func = ACTIVATE_FUNC[self.activate_func]

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

            self._b = b = self._bias_variable([hidden_dim], self.w_stddev)
            self._c = c = self._bias_variable([input_dim], self.w_stddev)

            encoded = activate_func(tf.matmul(X, W) + b)
            self._encoded = encoded

            Wt = tf.transpose(W)
            reconstructed = activate_func(tf.matmul(encoded, Wt) + c)
            self._reconstructed = reconstructed

            regularizer = self.lambda_w * tf.nn.l2_loss(W)
            error = tf.nn.l2_loss(clean_X - reconstructed)
            self._loss = loss = (error + regularizer) / batch_size
            self._optimize = tf.train.GradientDescentOptimizer(lr).minimize(loss)

            #variables summary
            tf.scalar_summary("l2_loss", loss)
            self._summ = tf.merge_all_summaries()

            #create session
            self._session = tf.Session(self.tf_master,
                                                config=tf.ConfigProto(
                                                    inter_op_parallelism_threads=self.num_cores,
                                                    intra_op_parallelism_threads=self.num_cores))
            #create initializer
            self._initializer = tf.initialize_all_variables()


    @classmethod
    def _weight_variable(cls, shape, stddev):
        init = tf.truncated_normal(shape, stddev)
        w = tf.Variable(init)
        return w

    @classmethod
    def _bias_variable(cls, shape, stddev):
        init = tf.truncated_normal(shape, stddev)
        b = tf.Variable(init)
        return b

    @classmethod
    def _generate_noise(cls, shape, stddev):
        noise = np.random.normal(size=shape, scale=stddev)
        return noise

if __name__ == '__main__':
    ae = TFAutoEncoder(hidden_dim=3,  num_cores=8,
                                    logdir="testlog", steps=0)
    X = np.arange(50000).reshape(5000, 10)
    ae.fit(X)
    Y = ae.reconstruct(X)
    print(X)
    print(Y)
