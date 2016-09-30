# models.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow_util as tfu
import util

_SEED = 66478  # Set to None for random seed.
_RANDOM = None

def get_fixed_random():
    global _RANDOM
    if _RANDOM is None:
        _RANDOM = util.FixedRandom()
    return _RANDOM

def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32, fixed_random=None):
    """
    Outputs random values from a truncated normal distribution.

    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than 2
    standard deviations from the mean are dropped and re-picked.
    """
    if fixed_random is None:
        fixed_random = get_fixed_random()
    value = np.empty(shape, dtype=dtype)
    for v in np.nditer(value, op_flags=['readwrite']):
        new_v = None
        while new_v is None or abs(new_v - mean) > 2 * abs(stddev):
            new_v = fixed_random.random.normalvariate(mean, stddev)
        v[...] = new_v
    return value


class Model(object):
    def __init__(self, input_ph=None, prediction_tensor=None, max_eval_batch_size=500):
        self.input_ph = input_ph
        self.prediction_tensor = prediction_tensor
        self._param_vars = OrderedDict()
        self._fixed_random = util.FixedRandom()  # deterministically initialize weights
        self._max_eval_batch_size = max_eval_batch_size

    @property
    def input_shape(self):
        input_shape = tuple(self.input_ph.get_shape().as_list()[1:])  # discard leading dimension (batch size)
        if None in input_shape:
            raise ValueError("the shape of the input_phs should be defined with the except of the leading dimension")
        return input_shape

    def add_param_var(self, param_var, name=None, **tags):
        if not isinstance(param_var, tf.Variable):
            param_var = tf.Variable(param_var, name=name)
        # parameters are trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        self._param_vars[param_var] = set(tag for tag, value in tags.items() if value)
        return param_var

    def get_param_vars(self, **tags):
        """
        Modified from here: https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/base.py
        """
        result = list(self._param_vars.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param_var for param_var in result
                      if not (only - self._param_vars[param_var])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param_var for param_var in result
                      if not (self._param_vars[param_var] & exclude)]
        return result

    def get_param_values(self, **tags):
        param_vars = self.get_param_vars(**tags)
        return [param_var.eval(session=tfu.get_session()) for param_var in param_vars]

    def set_param_values(self, param_values, **tags):
        param_vars = self.get_param_vars(**tags)
        if len(param_values) != len(param_vars):
            raise ValueError('there are %d parameter variables with the given tags'
                             'but %d parameter values were given' % (len(param_vars), len(param_values)))
        tfu.get_session().run([tf.assign(param_var, param_value) for (param_var, param_value) in zip(param_vars, param_values)])

    def get_batch_size(self, input_):
        if input_.shape == self.input_shape:  # input data is not batched
            batch_size = 0
        elif input_.shape[1:] == self.input_shape:
            batch_size = input_.shape[0]
        else:
            raise ValueError('expecting input of shape %r or %r but got input of shape %r' %
                             (self.input_shape, (None,) + self.input_shape, input_.shape))
        return batch_size

    def predict(self, input_):
        batch_size = self.get_batch_size(input_)
        if batch_size == 0:
            input_ = input_[None, :]
        # do the computation in smaller chunks because some GPUs don't have too much memory
        # the following block of code is equivalent to this line
        # prediction = self.prediction_tensor.eval(session=tfu.get_session(), feed_dict=dict([(self.input_ph, input_)]))
        predictions = []
        for i in range(0, batch_size, self._max_eval_batch_size):
            excerpt = slice(i, min(i+self._max_eval_batch_size, batch_size))
            prediction = self.prediction_tensor.eval(session=tfu.get_session(),
                                                     feed_dict=dict([(self.input_ph, input_[excerpt])]))
            predictions.append(prediction)
        prediction = np.concatenate(predictions, axis=0)
        if batch_size == 0:
            prediction = np.squeeze(prediction, axis=0)
        return prediction


class LinearRegressionModel(Model):
    def __init__(self, num_features=784, num_labels=10):
        super(LinearRegressionModel, self).__init__()
        # input and target placeholder variables
        self.x = tf.placeholder(tf.float32, shape=(None, num_features))
        self.input_ph = self.x

        # parameter variables
        self.W = self.add_param_var(truncated_normal([num_features, num_labels], stddev=0.1, fixed_random=self._fixed_random), name='W')
        self.b = self.add_param_var(tf.constant(0.1, shape=[num_labels]), name='b', regularizable=False)

        # prediction tensor
        self.y = tf.matmul(self.x, self.W) + self.b
        self.prediction_tensor = self.y

        # initialize parameters
        tfu.get_session().run([param_var.initializer for param_var in self.get_param_vars()])


class ClassifierModel(Model):
    def classify(self, input_datum_or_data):
        """
        Classifies a datum or each datum in a list of data.

        Args:
            input_datum_or_data: a 1-dimensional np.array of a single datum or
                a 2-dimensional np.array of data where each row is a datum.

        Returns:
            An integer (representing a label) if a single datum is passed in, or
                a list of integers (representing the labels) if multiple data
                is passed in.
        """
        prediction = self.predict(input_datum_or_data)
        category = np.argmax(prediction, axis=-1)
        return category

    def accuracy(self, input_data, target_data):
        """
        Computes the accuracy of the model classification predictions.

        Args:
            input_data: a 2-dimensional np.array of input data where each row is
                a datum.
            target_data: a 2-dimensional np.array of correct labels where each
                row is a probability distribution over the labels (or
                alternatively, a one-hot vector representation of the label).

        Returns:
            A float, the accuracy of the model for the given data.
        """
        category_labels = np.argmax(target_data, axis=-1)
        correct_prediction = self.classify(input_data) == category_labels
        accuracy = correct_prediction.mean()
        return accuracy


class SoftmaxRegressionModel(ClassifierModel):
    def __init__(self, num_features=784, num_labels=10):
        super(SoftmaxRegressionModel, self).__init__()
        # input and target placeholder variables
        self.x = tf.placeholder(tf.float32, shape=(None, num_features))
        self.input_ph = self.x

        # parameter variables
        self.W = self.add_param_var(truncated_normal([num_features, num_labels], stddev=0.1, fixed_random=self._fixed_random), name='W')
        self.b = self.add_param_var(tf.constant(0.1, shape=[num_labels]), name='b', regularizable=False)

        # prediction tensor
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.prediction_tensor = self.y

        # initialize parameters
        tfu.get_session().run([param_var.initializer for param_var in self.get_param_vars()])



class ConvNetModel(ClassifierModel):
    def __init__(self, use_batchnorm=False, use_dropout=False, x_shape=(None, 28, 28, 1), num_labels=10):
        super(ConvNetModel, self).__init__()
        _, image_size, _, num_channels = x_shape
        assert x_shape[2] == image_size
        self.x = tf.placeholder(tf.float32, shape=x_shape)
        self.input_ph = self.x
        is_train = True
        init_symmetry = False
        var_eps = 1e-20
        use_global_bn = True
        if use_global_bn:
            bn_axes = [0,1,2]
        else:
            bn_axes = [0]

        if init_symmetry:
            conv1_weights = tf.Variable(
                tf.zeros([5, 5, num_channels, 32]))  # 5x5 filter, depth 32.
            conv1_biases = tf.Variable(tf.zeros([32]))
            conv2_weights = tf.Variable(
               tf.zeros([5, 5, 32, 64]))
            conv2_biases = tf.Variable(tf.zeros([64]))
            fc1_weights = tf.Variable(  # fully connected, depth 512.
                tf.constant(0.1,
                      shape = [image_size // 4 * image_size // 4 * 64, 512],
                      ))
            fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
            fc2_weights = tf.Variable(
                tf.constant(0.1,shape=[512, num_labels],
                                  ))
            fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))
        else:
            conv1_weights = tf.Variable(
                truncated_normal([5, 5, num_channels, 32],  # 5x5 filter, depth 32.
                                      stddev=0.1, fixed_random=self._fixed_random))
            conv1_biases = tf.Variable(tf.zeros([32]))
            conv2_weights = tf.Variable(
               truncated_normal([5, 5, 32, 64],
                                  stddev=0.1, fixed_random=self._fixed_random))
            conv2_biases = tf.Variable(tf.constant(0., shape=[64]))
            fc1_weights = tf.Variable(  # fully connected, depth 512.
                truncated_normal(
                      [image_size // 4 * image_size // 4 * 64, 512],
                      stddev=0.1, fixed_random=self._fixed_random))
            fc1_biases = tf.Variable(tf.constant(0., shape=[512]))
            fc2_weights = tf.Variable(
                truncated_normal([512, num_labels],
                                  stddev=0.1, fixed_random=self._fixed_random))
            fc2_biases = tf.Variable(tf.constant(0., shape=[num_labels]))


        # Add parameter variables for solvers
        self.conv1_weights = self.add_param_var(conv1_weights)
        self.conv1_biases = self.add_param_var(conv1_biases)
        self.conv2_weights = self.add_param_var(conv2_weights)
        self.conv2_biases = self.add_param_var(conv2_biases)
        self.fc1_weights = self.add_param_var(fc1_weights)
        self.fc1_biases = self.add_param_var(fc1_biases)
        self.fc2_weights = self.add_param_var(fc2_weights)
        self.fc2_biases = self.add_param_var(fc2_biases)

        #Run Inference
        conv = tf.nn.conv2d(self.x,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

        conv = tf.nn.bias_add(conv, conv1_biases)

        #Add batch norm
        if use_batchnorm:
            mean,variance = tf.nn.moments(conv, bn_axes)
            conv = tf.nn.batch_normalization(conv, mean, variance, None,None,var_eps)

        relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        conv = tf.nn.bias_add(conv, conv2_biases)
        #Add batch norm
        if use_batchnorm:
            mean,variance = tf.nn.moments(conv, bn_axes)
            conv = tf.nn.batch_normalization(conv, mean, variance, None,None,var_eps)

        relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


        #Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
                pool,
                [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if is_train and use_dropout:
            hidden = tf.nn.dropout(hidden, 0.5, seed=_SEED)

        logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        self.prediction_tensor = tf.nn.softmax(logits)

        tfu.get_session().run([param_var.initializer for param_var in self.get_param_vars()])
