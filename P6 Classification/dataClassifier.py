# dataClassifier.py
# -----------------
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


import argparse
import numpy as np
import itertools
import perceptron
import models
import solvers
import datasets
import features
from search_hyperparams import search_hyperparams
import pacmanPlot
import graphicsUtils
from perceptron import PerceptronModel


def get_data(args):
    try:
        dataset = getattr(datasets, args.data)()
    except AttributeError:
        raise ValueError("Invalid option data %s" % args.data)
    if args.feature_extractor or args.model == 'ConvNetModel':
        # reshape each data point to be a square
        for i in range(0, len(dataset), 2):
            image_size = int(np.sqrt(dataset[i].shape[-1]))
            dataset[i] = dataset[i].reshape((-1, image_size, image_size, 1))
        if args.feature_extractor:
            print("Extracting features...")
            feature_extractor = getattr(features, args.feature_extractor)
            for i in range(0, len(dataset), 2):
                dataset[i] = np.array(map(feature_extractor, dataset[i]))
    train_data, val_data, test_data = dataset[:2], dataset[2:4], dataset[4:6]
    return train_data, val_data, test_data


def get_model_class(args):
    try:
        model_class = getattr(models, args.model)
    except AttributeError:
        try:
            model_class = getattr(perceptron, args.model)
        except AttributeError:
            raise ValueError("Invalid option model %s" % args.model)
    return model_class


def get_model(args, train_data):
    model_class = get_model_class(args)
    model_kwargs = dict(num_labels=train_data[1].shape[-1])
    if args.model == 'ConvNetModel':
        model_kwargs['x_shape'] = (None,) + train_data[0].shape[1:]
    else:
        model_kwargs['num_features'] = train_data[0].shape[-1]
    model = model_class(**model_kwargs)
    return model


def get_solver(args):
    try:
        solver_class = getattr(solvers, args.solver)
    except:
        try:
            solver_class = getattr(perceptron, args.solver)
        except AttributeError:
            raise ValueError("Invalid option solver %s" % args.solver)
    if solver_class == perceptron.PerceptronSolver:
        solver_kwargs = dict()
    elif solver_class == solvers.StochasticGradientDescentSolver:
        solver_kwargs = dict(learning_rate=args.learning_rate[0],
                             momentum=args.momentum[0],
                             weight_decay=args.weight_decay,
                             shuffle=not args.no_shuffle)
    elif solver_class == solvers.MinibatchStochasticGradientDescentSolver:
        solver_kwargs = dict(learning_rate=args.learning_rate[0],
                             momentum=args.momentum[0],
                             weight_decay=args.weight_decay,
                             batch_size=args.batch_size[0], shuffle=not args.no_shuffle)
    elif solver_class == solvers.GradientDescentSolver:
        solver_kwargs = dict(learning_rate=args.learning_rate[0],
                             momentum=args.momentum[0],
                             weight_decay=args.weight_decay)
    else:
        raise ValueError("Invalid option solver %s" % args.solver)
    if args.no_graphics:
        plot = 0
    else:
        plot = args.plot_interval
    solver = solver_class(iterations=args.iterations, plot=plot, **solver_kwargs)
    return solver


def pacman_display_callback(train_data):
    training = train_data[0]
    trainingLabels = train_data[1]
    if training.shape[1] == 2 and trainingLabels.shape[1] == 2:
        pacmanDisplay = pacmanPlot.PacmanPlotClassification2D();
        # plot points
        pacmanDisplay.plot(training, trainingLabels[:,0])
        graphicsUtils.sleep(0.1)

        # plot line
        def plot(model):
            weights = None
            if isinstance(model, PerceptronModel):
                weights = model.get_param_values()[0][:,0]
                updated = pacmanDisplay.setWeights(weights)
                if updated:
                    graphicsUtils.sleep(0.1)
            else:
                w1 = model.get_param_values()[0][:,0][:]
                b1 = model.get_param_values()[1][0]
                w2 = model.get_param_values()[0][:,1][:]
                b2 = model.get_param_values()[1][1]
                w = w1-w2
                b = b1-b2
                weights = np.r_[w,b]
                updated = pacmanDisplay.setWeights(weights)
                if updated:
                    graphicsUtils.sleep(0.1)
        return plot
    else:
        def do_nothing(model):
            pass
        return do_nothing


def main():
    model_choices = ['PerceptronModel', 'SoftmaxRegressionModel', 'ConvNetModel']
    solver_choices = ['PerceptronSolver', 'GradientDescentSolver', 'StochasticGradientDescentSolver', 'MinibatchStochasticGradientDescentSolver']
    data_choices = ['tinyMnistDataset', 'medMnistDataset', 'largeMnistDataset', 'mnistDataset', 'datasetA', 'datasetB']
    parser = argparse.ArgumentParser(description='Input the arguments to train the neural net.')
    parser.add_argument('-m', '--model', choices=model_choices, default='SoftmaxRegressionModel', help='Perceptron or neural net model')
    parser.add_argument('-s', '--solver', choices=solver_choices, default='MinibatchStochasticGradientDescentSolver', help='Solver to train the model')
    parser.add_argument('-d', '--data', choices=data_choices, default='medMnistDataset', help='Dataset to use for training',)
    parser.add_argument('-f', '--weight_file', default=None, help='File name (.npz) of weights to use to initialize the model')
    parser.add_argument('-i', '--iterations', default=10, type=int, help='Maximum iterations to run training')
    parser.add_argument('-l', '--learning_rate', nargs='+', default=[0.001], type=float, help='Learning rate to use for the solver')
    parser.add_argument('-b', '--batch_size', nargs='+', default=[32], type=int, help='Minibatch size to use when iterating the training and validation data')
    parser.add_argument('-u', '--momentum', nargs='+', default=[0.0], type=float, help='Momentum to use for the solver')
    parser.add_argument('-w', '--weight_decay', default=1e-3, type=float, help='Coefficient for l2 regularization on the loss')
    parser.add_argument('-bn', '--batch_norm', action='store_true', help='Batch normalization')
    parser.add_argument('--no-shuffle', action='store_true', help='Disables shuffling of data')
    parser.add_argument('--no-graphics', action='store_true', help='Turns off plots')
    parser.add_argument('-p', '--plot_interval', default=100, type=int, help='Only plot only every this often (in terms of iterations)')
    parser.add_argument('--print_features', action='store_true', help='Print high weight features')
    parser.add_argument('--feature_extractor', choices=['enhancedFeatureExtractor', 'basicFeatureExtractor'], help='Feature extractor function to use for mnist images')
    args = parser.parse_args()

    # Parse args and print information
    if args.model == 'PerceptronModel':
        args.solver = 'PerceptronSolver'
    print("data:\t\t" + args.data)
    print("model:\t\t" + args.model)
    print("solver:\t\t" + args.solver)

    train_data, val_data, test_data = get_data(args)

    # Load weights if applicable
    if args.weight_file is not None:
        print("loading parameter values from %s" % args.weight_file)
        init_param_values_file = np.load(args.weight_file)
        init_param_values = [init_param_values_file['arr_%d' % i] for i in
                             range(len(init_param_values_file.files))]
    else:
        init_param_values = None

    # train and validate
    hyperparams = [args.learning_rate, args.momentum, args.batch_size]
    if all([len(hyperparam) == 1 for hyperparam in hyperparams]):  # train and validate using a single set of hyperparameters
        model = get_model(args, train_data)
        if init_param_values is not None:
            model.set_param_values(init_param_values)
        solver = get_solver(args)
        print("Training...")
        solver.solve(*(train_data + val_data + [model, pacman_display_callback(train_data)]))
    else:  # do hyperparameter search
        # cartesian product of hyperparameters
        hyperparams = list(itertools.product(*hyperparams))
        model, best_hyperparams = search_hyperparams(*(train_data + val_data + zip(*hyperparams)), iterations=args.iterations,
                                                     model_class=get_model_class(args), init_param_values=init_param_values, use_bn=args.batch_norm)
        print('Best model is trained with these hyperparameters: learning_rate=%r, momentum=%r, batch_size=%r' % tuple(best_hyperparams))

    if args.print_features and args.model == 'PerceptronModel' and 'mnist' in args.data.lower():
        for l in model.legal_labels:
            highest_weighted_features = model.find_high_weight_features(l)
            features.print_features(highest_weighted_features)

    if 'mnist' in args.data.lower() and args.feature_extractor is not None:
        def get_data_labels_pred(data):
            features = data[0]
            labels = np.argmax(data[1], axis=-1)
            predictions = model.classify(features)
            return features, labels, predictions

        trainData, trainLabels, trainPredictions = get_data_labels_pred(train_data)
        validationData, validationLabels, validationPredictions = get_data_labels_pred(val_data)
        features.analysis(model, trainData, trainLabels, trainPredictions, validationData, validationLabels, validationPredictions)

    print("Computing accuracies")
    if train_data[0].shape[0] <= 10000:  # compute training accuracy only for small datasets (otherwise computing this is too slow)
        print("Train accuracy: %.1f%%" % (100.0 * model.accuracy(*train_data)))
    print("Validation accuracy: %.1f%%" % (100.0 * model.accuracy(*val_data)))
    print("Test accuracy: %.1f%%" % (100.0 * model.accuracy(*test_data)))
    raw_input('Press enter to exit')


if __name__ == '__main__':
    main()
