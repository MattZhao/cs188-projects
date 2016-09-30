# perceptron.py
# -------------
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


# Perceptron implementation
import util
import numpy as np
import models
import solvers
PRINT = True


class PerceptronClassifier(object):
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, num_features=None, num_labels=None):
        self.num_features = num_features or 784
        self.num_labels = num_labels or 10
        self.legal_labels = list(range(self.num_labels))
        self.weights = {}
        for l in self.legal_labels:
            self.weights[l] = util.Counter()  # this is the data-structure you should use

    def set_weights(self, weights):
        if isinstance(weights, dict):
            raise ValueError('weights should be a dict with each value being a util.Counter')
        if len(weights) != self.num_labels:
            raise ValueError('weights should be of length %d, weights of length %d given' % (self.num_labels, len(weights)))
        self.weights = weights

    def train(self, input_train_data, label_train_data, input_val_data, label_val_data, iterations, callback=None):
        """
        Question 1: Implement the multi-class version of the perceptron algorithm

        Args:
            input_train_data: list of util.Counters
            label_train_data: list of integers (representing the labels) of the same length as input_train_data
            input_val_data: list of util.Counters
            label_val_data: list of integers (representing the labels) of the same length as input_val_data
            iterations: number of iterations to pass over all the dataset
            callback: callback function for plotting

        The training loop for the perceptron passes through the training data
        several times and updates the weight vector for each label based on
        classification errors. See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).

        You don't need to use the validation data (input_val_data, label_val_data)
        for this question, but it is provided in case you want to check the
        accuracy on the validation data.

        Useful method:
        self.classify(...)
        """

        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        for iteration in range(iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(input_train_data)):
                # the callback plots the line in the Pacman Plot
                if callback is not None: callback()
                "*** YOUR CODE HERE ***"
                max_label = self.classify(input_train_data[i])
                prime_label = label_train_data[i]
                if max_label != prime_label:
                    update = input_train_data[i]
                    self.weights[max_label] -= update
                    self.weights[prime_label] += update

    def classify(self, input_datum_or_data):
        """
        Classifies a datum or each datum in a list of data.

        Args:
            input_datum_or_data: a single util.Counter or a list of them, where
                each util.Counter is a datum.

        Returns:
            An integer (representing a label) if a single datum is passed in, or
                a list of integers (representing the labels) if a list of data
                is passed in.
        """
        if isinstance(input_datum_or_data, util.Counter):
            input_datum = input_datum_or_data
            vectors = util.Counter()
            for l in self.legal_labels:
                vectors[l] = self.weights[l] * input_datum
            category_label = vectors.argMax()
            return category_label
        elif isinstance(input_datum_or_data, (list, tuple)):
            input_data = input_datum_or_data
            category_labels = [self.classify(input_datum) for input_datum in input_data]
            return category_labels
        else:
            raise ValueError("input_datum_or_data should be a util.Counter, "
                             "list or tuple, but a %r was given" % input_datum_or_data)

    def accuracy(self, input_data, label_data):
        predictions = self.classify(input_data)
        accuracy_count = [predictions[i] == label_data[i] for i in range(len(label_data))].count(True)
        return 1.0*accuracy_count / len(label_data)

    def find_high_weight_features(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        "*** YOUR CODE HERE ***"
        counter = self.weights[label].sortedKeys()
        best100Features = counter[0:100]
        return best100Features

class PerceptronModel(PerceptronClassifier, models.ClassifierModel):
    def get_param_values(self):
        param = np.empty((self.num_features, self.num_labels))
        for l in range(self.num_labels):
            for f in range(self.num_features):
                param[f, l] = self.weights[l][f]
        return [param]

    def set_param_values(self, params):
        try:
            param, = params
        except ValueError:
            raise ValueError('PerceptronModel only has one parameter, % parameters given' % len(params))
        if param.shape != (self.num_features, self.num_labels):
            raise ValueError('parameter should have shape %r, parameter with shape %r given' % ((self.num_features, self.num_labels), param.shape))
        for l in range(self.num_labels):
            for f in range(self.num_features):
                self.weights[l][f] = param[f, l]

    def classify(self, input_data):
        if isinstance(input_data, np.ndarray):
            input_data = util.counters_from_numpy_array(input_data)
        return PerceptronClassifier.classify(self, input_data)

    def accuracy(self, input_data, target_data):
        if isinstance(input_data, np.ndarray):
            input_data = util.counters_from_numpy_array(input_data)
        if isinstance(target_data, np.ndarray):
            target_data = util.list_from_numpy_array_one_hot(target_data)
        return PerceptronClassifier.accuracy(self, input_data, target_data)


class PerceptronSolver(solvers.Solver):
    def __init__(self, iterations, plot=0):
        self.iterations = iterations
        self.plot = plot

    def solve(self, input_train_data, target_train_data, input_val_data, target_val_data, model, callback=None):
        if not isinstance(model, PerceptronModel):
            raise ValueError('PerceptronSolver can only solve for PerceptronModel')
        # convert numpy arrays to util.Counters and lists
        print("Converting numpy arrays to counters and lists...")
        rows = input_train_data.shape[0]
        input_train_data = np.c_[input_train_data, np.ones((rows,1))]
        input_train_data = util.counters_from_numpy_array(input_train_data)
        target_train_data = util.list_from_numpy_array_one_hot(target_train_data)

        rows = input_val_data.shape[0]
        input_val_data = np.c_[input_val_data, np.ones((rows,1))]
        input_val_data = util.counters_from_numpy_array(input_val_data)
        target_val_data = util.list_from_numpy_array_one_hot(target_val_data)
        print("... done")

        if callback is None or self.plot == 0:
            train_callback = None
        else:
            train_callback = lambda: callback(model)

        model.train(input_train_data, target_train_data,
                    input_val_data, target_val_data,
                    iterations=self.iterations, callback=train_callback)
