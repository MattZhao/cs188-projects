# solvers.py
# ----------
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


import numpy as np
import tensorflow as tf
import tensorflow_util as tfu
from tensorflow_util import MinibatchIndefinitelyGenerator
import plotUtil
import util


def categorical_crossentropy(predictions, targets):
    return tf.reduce_mean(-tf.reduce_sum(targets * tf.log(tf.clip_by_value(predictions, 1e-10, float('inf'))), reduction_indices=[1]))

def squared_error(predictions, targets):
    return tf.reduce_mean(tf.reduce_sum(tf.square(targets - predictions), reduction_indices=[1]))


class Solver(object):
    """
    Solver abstract class.
    """
    def solve(self, input_train_data, target_train_data, input_val_data, target_val_data, model, callback=None):
        raise NotImplementedError


class GradientDescentSolver(Solver):
    def __init__(self, learning_rate, iterations, momentum=0, weight_decay=1e-3, loss_function=None, plot=0):
        """
        Gradient descent solver for optimizing a model given some data.

        Args:
            learning_rate: also known as alpha. Used for parameter updates.
            iterations: number of gradient steps (i.e. updates) to perform when
                solving a model.
            momentum: also known as mu. Used for velocity updates.
            weight_decay: coefficient for l2 regularization on the loss.
            loss_function: loss function to use for the objective being optimized.
            plot: whether to show a plot of the loss for every iteration.
        """
        if learning_rate < 0:
            raise ValueError('learning_rate should be a non-negative number, %r given' % learning_rate)
        self.learning_rate = learning_rate
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError('iterations should be a non-negative integer, %d given' % iterations)
        self.iterations = iterations
        if not (0 <= momentum <= 1):
            raise ValueError('momentum should be between 0 and 1 (inclusive), %r given' % momentum)
        self.momentum = momentum
        if weight_decay < 0:
            raise ValueError('weight_decay should be a non-negative number, %r given' % weight_decay)
        self.weight_decay = weight_decay
        self.loss_function = loss_function or categorical_crossentropy
        self.plot = plot

    def get_updates_without_momentum(self, loss_tensor, param_vars):
        """
        Question 4: Returns the gradient descent updates when no momentum is used.

        Args:
            loss_tensor: loss tensor used to compute the gradients.
            param_vars: list of parameter variables.

        Returns:
            A list of tuples, where each tuple is an update of the form
            (param_var, new_param_tensor) indicating that, at runtime, the
            parameter param_var should be updated with new_param_tensor.

        You implementation should use the gradient tensors (provided below)
        and the member variable self.learning_rate.
        """
        grad_tensors = tf.gradients(loss_tensor, param_vars)
        updates = []
        "*** YOUR CODE HERE ***"
        for i in range(len(param_vars)):
            param_var = param_vars[i]
            new_param_tensor = param_var - self.learning_rate * grad_tensors[i]
            updates.append((param_var, new_param_tensor))
        return updates

    def get_updates_with_momentum(self, loss_tensor, param_vars):
        """
        Question 5: Returns the gradient descent updates when momentum is used.

        Args:
            loss_tensor: loss tensor used to compute the gradients.
            param_vars: list of parameter variables.

        Returns:
            A list of tuples, where each tuple is an update of the form 
            (var, new_tensor) indicating that, at runtime, the variable var
            should be updated with new_tensor.

        You implementation should use the gradient tensors and the velocity
        variables (both provided below), and the member variables
        self.learning_rate and self.momentum.
        """
        grad_tensors = tf.gradients(loss_tensor, param_vars)
        vel_vars = [tf.Variable(np.zeros(param_var.get_shape(), dtype=np.float32)) for param_var in param_vars]
        tfu.get_session().run([vel_var.initializer for vel_var in vel_vars])
        updates = []
        "*** YOUR CODE HERE ***"
        for i in range(len(param_vars)):
            new_vel = self.momentum * vel_vars[i] - self.learning_rate * grad_tensors[i]
            updates.append((vel_vars[i], new_vel))
            new_tensor = param_vars[i] + new_vel
            updates.append((param_vars[i], new_tensor))
        return updates

    def get_loss_tensor(self, prediction_tensor, target_ph, param_vars):
        loss_tensor = self.loss_function(prediction_tensor, target_ph)
        loss_tensor += self.weight_decay * sum(tf.nn.l2_loss(param_var) for param_var in param_vars)
        return loss_tensor

    def get_updates(self, loss_tensor, param_vars):
        """
        Returns the gradient descent updates.

        Args:
            loss_tensor: loss tensor used to compute the gradients.
            param_vars: list of parameter variables.

        Returns:
            A list of tuples, where each tuple is an update of the form
            (var, new_tensor) indicating that, at runtime, the variable var
            should be updated with new_tensor.
        """
        if self.momentum == 0:
            return self.get_updates_without_momentum(loss_tensor, param_vars)
        else:
            return self.get_updates_with_momentum(loss_tensor, param_vars)

    def solve(self, input_train_data, target_train_data, input_val_data, target_val_data, model, callback=None):
        """
        Question 6.a: Optimize the model and return the intermediate losses.

        Optimize the model using gradient descent by running the variable
        updates for self.iterations iterations.

        Args:
            input_train_data: a numpy.array with shape (N, R)
            target_train_data: a numpy.array with shape (N, S)
            input_val_data: a numpy.array with shape (M, R)
            target_val_data: a numpy.array with shape (M, S)
            model: the model from which the parameters are optimized

        Returns:
            A tuple of lists, where the first list contains the training loss of
            each iteration and the second list contains the validation loss of
            each iteration.

        N and M are the numbers of training points, respectively, and R and S
        are the dimensions for each input and target data point, respectively.

        You may not need to fill in both "*** YOUR CODE HERE ***" blanks,
        but they are both provided so you can define variables outside and
        inside the for loop.

        Useful method:
        session.run
        """
        session = tfu.get_session()
        target_ph = tf.placeholder(tf.float32, shape=(None,) + target_train_data.shape[1:])
        placeholders = [model.input_ph, target_ph]
        train_data = [input_train_data, target_train_data]
        val_data = [input_val_data, target_val_data]
        # You may want to initialize some variables that are shared across iterations
        "*** YOUR CODE HERE ***"
        loss_tensor = self.get_loss_tensor(model.prediction_tensor, target_ph, model.get_param_vars(regularizable=True))
        updates = self.get_updates(loss_tensor, model.get_param_vars(trainable=True))
        update_ops = [tf.assign(old_var, new_var_or_tensor) for (old_var, new_var_or_tensor) in updates]
        train_losses = []
        val_losses = []
        for iter_ in range(self.iterations):
            "*** YOUR CODE HERE ***"
            # train_loss should be the loss of this iteration using all of the training data
            # val_loss should be the loss of this iteration using all of the validation data
            train_loss = session.run(loss_tensor, feed_dict = {model.input_ph: input_train_data, target_ph: target_train_data})
            session.run(update_ops, feed_dict = {model.input_ph: input_train_data, target_ph: target_train_data})
            val_loss = session.run(loss_tensor, feed_dict = {model.input_ph: input_val_data, target_ph: target_val_data})

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if callback is not None: callback(model)
            self.display_progress(iter_, train_losses, val_losses)
        return train_losses, val_losses

    def display_progress(self, iter_, train_losses, val_losses):
        print("Iteration {} of {}".format(iter_, self.iterations))
        print("    training loss = {:.6f}".format(train_losses[-1]))
        print("    validation loss = {:.6f}".format(val_losses[-1]))
        if self.plot and iter_ % self.plot == 0:
            plotUtil.plotTwoCurves(range(len(train_losses)), train_losses,
                                   range(len(val_losses)), val_losses,
                                   label1='training loss',
                                   label2='validation loss', showLegend=True,
                                   figureIdx=2,
                                   figureTitle="%s: Training and Validation Loss" % self.__class__.__name__)


class StochasticGradientDescentSolver(GradientDescentSolver):
    def __init__(self, learning_rate, iterations, momentum=0, weight_decay=1e-3, shuffle=None, loss_function=None, plot=0):
        """
        Stochastic gradient descent solver for optimizing a model given some data.

        Args:
            learning_rate: also known as alpha. Used for parameter updates.
            iterations: number of gradient steps (i.e. updates) to perform when
                solving a model.
            momentum: also known as mu. Used for velocity updates.
            weight_decay: coefficient for l2 regularization on the loss.
            shuffle: whether the order of the data points should be randomized
                when iterating over the data
            loss_function: loss function to use for the objective being optimized.
            plot: whether to show a plot of the loss for every iteration.
        """
        super(StochasticGradientDescentSolver, self).__init__(
            learning_rate, iterations, momentum=momentum, weight_decay=weight_decay, loss_function=loss_function, plot=plot)
        self.shuffle = True if shuffle is None else shuffle

    def solve(self, input_train_data, target_train_data, input_val_data, target_val_data, model, callback=None):
        """
        Question 6.b: Optimize the model and return the intermediate losses.

        Optimize the model using stochastic gradient descent by running the
        variable updates for self.iterations iterations.

        Args:
            input_train_data: a numpy.array with shape (N, R)
            target_train_data: a numpy.array with shape (N, S)
            input_val_data: a numpy.array with shape (M, R)
            target_val_data: a numpy.array with shape (M, S)
            model: the model from which the parameters are optimized

        Returns:
            A tuple of lists, where the first list contains the training loss of
            each iteration and the second list contains the validation loss of
            each iteration. The validation loss should be computed using the
            same amount of data as the training loss, but using the validation
            data.

        N and M are the numbers of training points, respectively, and R and S
        are the dimensions for each input and target data point, respectively.

        In here, the gradient descent is stochastic, meaning that you don't
        need to use all the data at once before you update the model
        parameters. Instead, you update the model parameters as you iterate
        over the data. You must use MinibatchIndefinitelyGenerator to iterate
        over the data, otherwise your solution might differ from the one of
        the autograder. You will need to instantiate two generators (one for
        the training data and another one for the validation data) and you
        should do it before the for loop. You should read the docstring of
        MinibatchIndefinitelyGenerator in tensorflow_util.py to figure out
        how to use it. Make sure to pass in self.shuffle when you instantiate
        the generator. You will have to choose a proper batch size too.

        Useful member variables and methods:
        self.shuffle
        session.run(...)
        generator.next()
        """
        session = tfu.get_session()
        target_ph = tf.placeholder(tf.float32, shape=(None,) + target_train_data.shape[1:])
        placeholders = [model.input_ph, target_ph]
        train_data = [input_train_data, target_train_data]
        val_data = [input_val_data, target_val_data]
        # You may want to initialize some variables that are shared across iterations
        "*** YOUR CODE HERE ***"
        train_gen = MinibatchIndefinitelyGenerator(train_data, 1, self.shuffle)
        val_gen = MinibatchIndefinitelyGenerator(val_data, 1, self.shuffle)

        loss_tensor = self.get_loss_tensor(model.prediction_tensor, target_ph, model.get_param_vars(regularizable=True))
        updates = self.get_updates(loss_tensor, model.get_param_vars(trainable=True))
        update_ops = [tf.assign(old_var, new_var_or_tensor) for (old_var, new_var_or_tensor) in updates]
        train_losses = []
        val_losses = []
        for iter_ in range(self.iterations):
            "*** YOUR CODE HERE ***"
            # train_loss should be the loss of this iteration using only the training data that was used for the updates
            # val_loss should be the loss of this iteration using the same amount of data used for the updates, but using the validation data instead
            a, b = train_gen.next()
            train_loss = session.run(loss_tensor, feed_dict = {model.input_ph: a, target_ph: b})
            session.run(update_ops, feed_dict = {model.input_ph: a, target_ph: b})
            c, d = val_gen.next()
            val_loss = session.run(loss_tensor, feed_dict = {model.input_ph: c, target_ph: d})

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if callback is not None: callback(model)
            self.display_progress(iter_, train_losses, val_losses)
        return train_losses, val_losses


class MinibatchStochasticGradientDescentSolver(GradientDescentSolver):
    def __init__(self, learning_rate, iterations, batch_size, momentum=0, weight_decay=1e-3, shuffle=None, loss_function=None, plot=0):
        """
        Minibatch stochastic gradient descent solver for optimizing a model given some data.

        Args:
            learning_rate: also known as alpha. Used for parameter updates.
            iterations: number of gradient steps (i.e. updates) to perform when
                solving a model.
            batch_size: minibatch size to use when iterating the training and
                validation data.
            momentum: also known as mu. Used for velocity updates.
            weight_decay: coefficient for l2 regularization on the loss.
            shuffle: whether the order of the data points should be randomized
                when iterating over the data
            loss_function: loss function to use for the objective being optimized.
            plot: whether to show a plot of the loss for every iteration.
        """
        super(MinibatchStochasticGradientDescentSolver, self).__init__(
            learning_rate, iterations, momentum=momentum, weight_decay=weight_decay, loss_function=loss_function, plot=plot)
        self.shuffle = True if shuffle is None else shuffle
        if not isinstance(batch_size, int) or batch_size < 0:
            raise ValueError('batch_size should be a non-negative integer, %d given' % batch_size)
        self.batch_size = batch_size

    def solve(self, input_train_data, target_train_data, input_val_data, target_val_data, model, callback=None):
        """
        Question 6.c: Optimize the model and return the intermediate losses.

        Optimize the model using minibatch stochastic gradient descent by
        running the variable updates for self.iterations iterations.

        Args:
            input_train_data: a numpy.array with shape (N, R)
            target_train_data: a numpy.array with shape (N, S)
            input_val_data: a numpy.array with shape (M, R)
            target_val_data: a numpy.array with shape (M, S)
            model: the model from which the parameters are optimized

        Returns:
            A tuple of lists, where the first list contains the training loss of
            each iteration and the second list contains the validation loss of
            each iteration. The validation loss should be computed using the
            same amount of data as the training loss, but using the validation
            data.

        N and M are the numbers of training points, respectively, and R and S
        are the dimensions for each input and target data point, respectively.

        For minibatch stochastic gradient descent, you will need to iterate
        over the data in minibatches. As before, you must use
        MinibatchIndefinitelyGenerator to iterate over the data. You will
        need to instantiate two generators (one for the training data and
        another one for the validation data) and you should do it before the
        for loop. You should read the docstring of
        MinibatchIndefinitelyGenerator in tensorflow_util.py to figure out
        how to use it. Make sure to pass in self.batch_size and self.shuffle
        when you instantiate the generator.

        Useful member variables and methods:
        self.batch_size
        self.shuffle
        session.run(...)
        generator.next()
        """
        session = tfu.get_session()
        target_ph = tf.placeholder(tf.float32, shape=(None,) + target_train_data.shape[1:])
        placeholders = [model.input_ph, target_ph]
        train_data = [input_train_data, target_train_data]
        val_data = [input_val_data, target_val_data]
        # You may want to initialize some variables that are shared across iterations
        "*** YOUR CODE HERE ***"
        train_gen = MinibatchIndefinitelyGenerator(train_data, self.batch_size, self.shuffle)
        val_gen = MinibatchIndefinitelyGenerator(val_data, self.batch_size, self.shuffle)

        loss_tensor = self.get_loss_tensor(model.prediction_tensor, target_ph, model.get_param_vars(regularizable=True))
        updates = self.get_updates(loss_tensor, model.get_param_vars(trainable=True))
        update_ops = [tf.assign(old_var, new_var_or_tensor) for (old_var, new_var_or_tensor) in updates]
        train_losses = []
        val_losses = []
        for iter_ in range(self.iterations):
            "*** YOUR CODE HERE ***"
            a, b = train_gen.next()
            train_loss = session.run(loss_tensor, feed_dict = {model.input_ph: a, target_ph: b})
            session.run(update_ops, feed_dict = {model.input_ph: a, target_ph: b})
            c, d = val_gen.next()
            val_loss = session.run(loss_tensor, feed_dict = {model.input_ph: c, target_ph: d})
            
            # train_loss should be the loss of this iteration using only the training data that was used for the updates
            # val_loss should be the loss of this iteration using the same amount of data used for the updates, but using the validation data instead
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if callback is not None: callback(model)
            self.display_progress(iter_, train_losses, val_losses)
        return train_losses, val_losses
