# classificationTestClasses.py
# ----------------------------
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


from hashlib import sha1
import testClasses
import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import models
import solvers
import datasets
import tensorflow_util as tfu
from search_hyperparams import search_hyperparams
import answers


def tinyLinearRegressionModelInitParamValues():
    W_init = np.array([[0.4, 0.0], [0.0, -0.2], [0.1, 0.0]], dtype=np.float32)
    b_init = np.array([-0.5, 0.3], dtype=np.float32)
    return [W_init, b_init]

def mnistSoftmaxRegressionModel():
    return models.SoftmaxRegressionModel()

def mnistConvNetModel(use_batchnorm, use_dropout):
    return models.ConvNetModel(use_batchnorm, use_dropout)

class UpdatesEqualityTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(UpdatesEqualityTest, self).__init__(question, testDict)
        # solver
        self.solver_module = testDict['solver_module']
        self.solver_class = testDict['solver_class']
        self.learning_rate = float(testDict['learning_rate'])
        self.momentum = float(testDict.get('momentum', 0))
        # model
        self.model_module = testDict.get('model_module', 'models')
        self.model_class = testDict['model_class']
        self.init_param_values_fname_or_function = testDict.get('init_param_values_fname_or_function', None)
        # dataset
        self.dataset = testDict['dataset']
        # other
        self.update_iterations = int(testDict.get('update_iterations', 0))
        self.max_points = int(testDict['max_points'])

    def get_update_values(self, moduleDict):
        # dataset
        dataset = getattr(datasets, self.dataset)()
        input_data, target_data = dataset[:2]
        # model
        if self.model_module == 'models':  # need to check for this since this is not a student file
            module = models
        else:
            module = moduleDict[self.model_module]
        model_class = getattr(module, self.model_class)
        model_kwargs = dict(num_labels=dataset[1].shape[-1])
        if self.model_class == 'ConvNetModel':
            model_kwargs['x_shape'] = (None,) + dataset[0].shape[1:]
        else:
            model_kwargs['num_features'] = dataset[0].shape[-1]
        model = model_class(**model_kwargs)
        # solver
        solver_class = getattr(moduleDict[self.solver_module], self.solver_class)
        solver = solver_class(learning_rate=self.learning_rate, iterations=0, momentum=self.momentum)

        target_ph = tf.placeholder(tf.float32, shape=(None, 2))
        loss_tensor = solvers.squared_error(model.prediction_tensor, target_ph)
        param_vars = model.get_param_vars(trainable=True)

        updates = solver.get_updates(loss_tensor, param_vars)
        update_ops = [tf.assign(old, new) for (old, new) in updates]
        feed_dict = dict(zip([model.input_ph, target_ph], [input_data, target_data]))
        for i in range(self.update_iterations):
            tfu.get_session().run(update_ops, feed_dict=feed_dict)

        grad_tensors = tf.gradients(loss_tensor, param_vars)
        grads = [grad_tensor.eval(session=tfu.get_session(), feed_dict=feed_dict) for grad_tensor in grad_tensors]

        len_messages = len(self.messages)
        if not isinstance(updates, (list, tuple)):
            self.addMessage('updates should be a list, %r given' % updates)
            return updates, None, grads
        # Check updates are in the right format
        for update in updates:
            try:
                old, new = update
            except ValueError:
                self.addMessage('Each update in updates should be of length 2, but it is of length %d' % len(update))
                continue
            if not isinstance(old, tf.Variable):
                self.addMessage('The first element in the tuple update should be a tf.Variable, %r given' % old)
            if not isinstance(new, (tf.Variable, tf.Tensor)):
                self.addMessage('The second element in the tuple update should be a tf.Variable or a tf.Tensor, %r given' % new)
        if len(self.messages) > len_messages:
            return updates, None, grads
        # Check for repeated variables
        if len(set(zip(*updates)[0])) != len(updates):
            self.addMessage('There are some repeated variables being updated: %r' % zip(*updates)[0])
            return updates, None, grads
        update_values = [tfu.get_session().run(update, feed_dict=feed_dict) for update in updates]
        return updates, update_values, grads

    def update_values_allclose(self, update_values, gold_update_values):
        if len(update_values) != len(gold_update_values):
            self.addMessage('Expecting %d update tuples, but %d were given' % (len(gold_update_values), len(update_values)))
            return False
        num_equal_updates = 0
        update_values = list(update_values)
        for gold_update_value in gold_update_values:
            for i, update_value in enumerate(update_values):
                allclose = all([value.shape == gold_value.shape and np.allclose(value, gold_value) for (value, gold_value) in zip(update_value, gold_update_value)])
                if allclose:
                    update_values.pop(i)
                    num_equal_updates += 1
                    break
        assert num_equal_updates <= len(gold_update_values)
        if num_equal_updates < len(gold_update_values):
            self.addMessage('Only %d out of %d update tuples are equal' % (num_equal_updates, len(gold_update_values)))
            return False
        return True

    def execute(self, grades, moduleDict, solutionDict):
        updates, update_values, grads = self.get_update_values(moduleDict)
        gold_update_values_file = np.load(solutionDict['update_values_fname'])
        gold_update_values = [gold_update_values_file['arr_%d' % i] for i in range(len(gold_update_values_file.files))]
        gold_update_values = zip(gold_update_values[::2], gold_update_values[1::2])
        correct = self.update_values_allclose(update_values, gold_update_values)
        if correct:
            total_points = self.max_points
            return self.testPartial(grades, total_points, self.max_points)

        hyperparameters_str = 'learning_rate=%d' % self.learning_rate
        if self.momentum != 0:
            hyperparameters_str += ', momentum=%d' % self.momentum
        self.addMessage('Update tuple values (with %s) are not equal.\n' % hyperparameters_str)
        self.addMessage('Gradients with respect to each parameter are:\n%r\n' % grads)
        self.addMessage('Student\'s evaluated update values are:\n%r\n' % update_values)
        self.addMessage('Correct evaluated update values are:\n%r\n' % gold_update_values)
        return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        updates, update_values, grads = self.get_update_values(moduleDict)
        path, fname = os.path.split(self.path)
        fname, ext = os.path.splitext(fname)
        fname = os.path.join(path, fname + '.npz')
        np.savez(fname, *[value for update_value in update_values for value in update_value])
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('update_values_fname: "%s"\n' % fname)
        return True



class ParamValuesEqualityTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(ParamValuesEqualityTest, self).__init__(question, testDict)
        # solver
        self.solver_module = testDict.get('solver_module', 'solvers')
        self.solver_class = testDict['solver_class']
        self.solver_kwargs = OrderedDict(iterations=int(testDict['iterations']))
        if 'learning_rate' in testDict:
            self.solver_kwargs['learning_rate'] = float(testDict['learning_rate'])
        if 'momentum' in testDict:
            self.solver_kwargs['momentum'] = float(testDict['momentum'])
        if 'shuffle' in testDict:
            self.solver_kwargs['shuffle'] = bool(testDict['shuffle'])
        if 'batch_size' in testDict:
            self.solver_kwargs['batch_size'] = int(testDict['batch_size'])
        if 'solver_hyperparams' in testDict:
            hyperparams_dict = getattr(answers, testDict['solver_hyperparams'])()
            # filter out any item in the dict that is not the learning rate nor momentum
            allowed_hyperparams = ['learning_rate', 'momentum']
            hyperparams_dict = dict([(k, v) for (k, v) in hyperparams_dict.items() if k in allowed_hyperparams])
            self.solver_kwargs.update(hyperparams_dict)
        if 'loss_function' in testDict:
            self.solver_kwargs['loss_function'] = getattr(solvers, self.loss_function)
        self.check_losses = bool(testDict.get('check_losses', False))
        # model
        self.model_module = testDict.get('model_module', 'models')
        self.model_class = testDict['model_class']
        self.init_param_values_fname_or_function = testDict.get('init_param_values_fname_or_function', None)
        self.batch_norm = bool(testDict.get('batch_norm', False))
        # dataset
        self.dataset = testDict['dataset']
        self.feature_extractor = testDict.get('feature_extractor', None)
        # other
        self.max_points = int(testDict['max_points'])

    def load_init_param_values(self, init_param_values_fname):
        init_param_values_file = np.load(init_param_values_fname)
        init_param_values = [init_param_values_file['arr_%d' % i] for i in range(len(init_param_values_file.files))]
        return init_param_values

    def get_solved_model_and_dataset(self, moduleDict, init_param_values=None):
        # dataset
        dataset = getattr(datasets, self.dataset)()
        if self.feature_extractor or self.model_class == 'ConvNetModel':
            # reshape each data point to be a square
            for i in range(0, len(dataset), 2):
                image_size = int(np.sqrt(dataset[i].shape[-1]))
                dataset[i] = dataset[i].reshape((-1, image_size, image_size, 1))
            if self.feature_extractor:
                import features
                feature_extractor = getattr(features, self.feature_extractor)
                for i in range(0, len(dataset), 2):
                    dataset[i] = np.array(map(feature_extractor, dataset[i]))
        # model
        if self.model_module == 'models':  # need to check for this since this is not a student file
            module = models
        else:
            module = moduleDict[self.model_module]
        model_class = getattr(module, self.model_class)
        model_kwargs = dict(num_labels=dataset[1].shape[-1])
        if self.model_class == 'ConvNetModel':
            model_kwargs['x_shape'] = (None,) + dataset[0].shape[1:]
            model_kwargs['use_batchnorm'] = self.batch_norm
        else:
            model_kwargs['num_features'] = dataset[0].shape[-1]
        model = model_class(**model_kwargs)
        if init_param_values is not None:
            model.set_param_values(init_param_values)
        else:
            init_param_values = model.get_param_values()
        # solver
        solver_class = getattr(moduleDict[self.solver_module], self.solver_class)
        solver = solver_class(**self.solver_kwargs)
        losses = solver.solve(*(dataset[:4] + [model]))
        return init_param_values, model, dataset, losses

    def execute(self, grades, moduleDict, solutionDict):
        if 'init_param_values_fname' in solutionDict:
            init_param_values = self.load_init_param_values(solutionDict['init_param_values_fname'])
        else:
            init_param_values = None
        init_param_values, model, dataset, losses = self.get_solved_model_and_dataset(moduleDict, init_param_values)
        param_values = model.get_param_values()
        gold_param_values = self.load_init_param_values(solutionDict['param_values_fname'])
        correct_param_values = all([np.allclose(param_value, gold_param_value, atol=1e-07) for (param_value, gold_param_value) in zip(param_values, gold_param_values)])
        if self.check_losses:
            gold_losses = self.load_init_param_values(solutionDict['losses_fname'])
            correct_losses = all([np.allclose(loss, gold_loss, atol=1e-07) for (loss, gold_loss) in zip(losses, gold_losses)])
        else:
            correct_losses = True
        if correct_param_values and correct_losses:
            total_points = self.max_points
            return self.testPartial(grades, total_points, self.max_points)
        hyperparameters_str = ', '.join(['%s=%r' % (k, v) for (k, v) in self.solver_kwargs.items()])
        if not correct_losses:
            self.addMessage('Intermediate losses from solving (with %s) are not equal.\n' % hyperparameters_str)
            self.addMessage('Student\'s losses from solving are:\ntraining loss: %r\nvalidation loss: %r\n' % (losses))
            self.addMessage('Correct losses from solving are:\ntraining loss: %r\nvalidation loss: %r\n' % tuple([loss.tolist() for loss in gold_losses]))
        if not correct_param_values:
            self.addMessage('Parameter values after solving (with %s) are not equal.\n' % hyperparameters_str)
            self.addMessage('Student\'s parameter values after solving are:\n%r\n' % param_values)
            self.addMessage('Correct parameter values after solving are:\n%r\n' % gold_param_values)
        return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        if self.init_param_values_fname_or_function is not None:
            if self.init_param_values_fname_or_function.endswith('.npz'):
                init_param_values = self.load_init_param_values(self.init_param_values_fname_or_function)
                init_param_values_fname = self.init_param_values_fname_or_function
            elif self.init_param_values_fname_or_function == '':
                init_param_values_fname = ''
            else:
                init_param_values = globals()[self.init_param_values_fname_or_function]()
                init_param_values_fname = None
        else:
            init_param_values = None
            init_param_values_fname = None
        init_param_values, model, dataset, losses = self.get_solved_model_and_dataset(moduleDict, init_param_values)
        # save init param values
        path, fname = os.path.split(self.path)
        fname, ext = os.path.splitext(fname)
        param_values_fname = os.path.join(path, fname + '.npz')
        # only save init param values if it doesn't already exist
        if init_param_values_fname is None:
            init_param_values_fname = os.path.join(path, fname + '_init.npz')
            np.savez(init_param_values_fname, *init_param_values)
        np.savez(param_values_fname, *model.get_param_values())
        # save losses if applicable
        if self.check_losses:
            losses_fname = os.path.join(path, fname + '_losses.npz')
            np.savez(losses_fname, *losses)
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            if init_param_values_fname:
                handle.write('init_param_values_fname: "%s"\n' % init_param_values_fname)
            handle.write('param_values_fname: "%s"\n' % param_values_fname)
            if self.check_losses:
                handle.write('losses_fname: "%s"\n' % losses_fname)
        return True



class GradeClassifierTest(ParamValuesEqualityTest):

    def __init__(self, question, testDict):
        super(GradeClassifierTest, self).__init__(question, testDict)
        self.accuracy_threshold = float(testDict['accuracy_threshold'])

    def grade_classifier(self, moduleDict, solutionDict):
        if 'init_param_values_fname' in solutionDict:
            init_param_values = self.load_init_param_values(solutionDict['init_param_values_fname'])
        else:
            init_param_values = None
        init_param_values, model, dataset, losses = self.get_solved_model_and_dataset(moduleDict, init_param_values)
        train_data, val_data, test_data = dataset[:2], dataset[2:4], dataset[4:]
        print("Computing accuracies")
        test_accuracy = model.accuracy(*test_data)
        if train_data[0].shape[0] <= 10000:  # compute training accuracy only for small datasets (otherwise computing this is too slow)
            print("Train accuracy: %.1f%%" % (100.0 * model.accuracy(*train_data)))
        print("Validation accuracy: %.1f%%" % (100.0 * model.accuracy(*val_data)))
        print("Test accuracy: %.1f%%" % (100.0 * test_accuracy))
        return test_accuracy

    def execute(self, grades, moduleDict, solutionDict):
        accuracy = 100.0 * self.grade_classifier(moduleDict, solutionDict)

        if accuracy >= self.accuracy_threshold:
            total_points = self.max_points
            return self.testPartial(grades, total_points, self.max_points)

        self.addMessage('Student\'s accuracy after solving is:\n%r\n' % accuracy)
        self.addMessage('Accuracy threshold to pass the test is:\n%r\n' % self.accuracy_threshold)
        return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        if self.init_param_values_fname_or_function is not None:
            if self.init_param_values_fname_or_function.endswith('.npz'):
                init_param_values = self.load_init_param_values(self.init_param_values_fname_or_function)
                init_param_values_fname = self.init_param_values_fname_or_function
            elif self.init_param_values_fname_or_function == '':
                init_param_values = None
                init_param_values_fname = ''
            else:
                init_param_values = globals()[self.init_param_values_fname_or_function]()
                init_param_values_fname = None
        else:
            init_param_values = None
            init_param_values_fname = None
        init_param_values, model, dataset, losses = self.get_solved_model_and_dataset(moduleDict, init_param_values)
        # only save init param values if it doesn't already exist
        if init_param_values_fname is None:
            path, fname = os.path.split(self.path)
            fname, ext = os.path.splitext(fname)
            init_param_values_fname = os.path.join(path, fname + '_init.npz')
            np.savez(init_param_values_fname, *init_param_values)
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            if init_param_values_fname:
                handle.write('init_param_values_fname: "%s"\n' % init_param_values_fname)
        return True


class HyperParamEqualityTest(testClasses.TestCase):
    def __init__(self, question, testDict):
        super(HyperParamEqualityTest, self).__init__(question, testDict)
        self.dataset = getattr(datasets, testDict['dataset'])()
        self.maxPoints = int(testDict['maxPoints'])
        self.learning_rates = [float(x) for x in testDict['learning_rates'].split()]
        self.momentums = [float(x) for x in testDict['momentums'].split()]
        self.batch_sizes = [int(x) for x in testDict['batch_sizes'].split()]
        self.model_class = models.SoftmaxRegressionModel
        self.iterations = int(testDict['iterations'])

    def hyperparam_exp_get_model(self, init_param_values):
        hyperparams = [self.learning_rates, self.momentums, self.batch_sizes]
        best_model, best_hyperparams = search_hyperparams(*(self.dataset[:4] + hyperparams),
                                                          iterations=self.iterations,
                                                          model_class=self.model_class,
                                                          init_param_values=init_param_values)
        return best_model.get_param_values()


    def execute(self, grades, moduleDict, solutionDict):
        init_param_values_file = np.load(solutionDict['init_param_values_fname'])
        init_param_values = [init_param_values_file['arr_%d' % i] for i in range(len(init_param_values_file.files))]
        param_values = self.hyperparam_exp_get_model(init_param_values)

        gold_param_values_file = np.load(solutionDict['param_values_fname'])
        gold_param_values = [gold_param_values_file['arr_%d' % i] for i in range(len(gold_param_values_file.files))]
        correct = all([np.allclose(new_param_value, gold_param_value, atol=1e-07) for (new_param_value, gold_param_value) in zip(param_values, gold_param_values)])
        if correct:
            totalPoints = self.maxPoints
            return self.testPartial(grades, totalPoints, self.maxPoints)

        self.addMessage('Student\'s parameter values of the best model are:\n%r\n' % param_values)
        self.addMessage('Correct parameter values of the model are:\n%r\n' % gold_param_values)
        return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        path, fname = os.path.split(self.path)
        fname, ext = os.path.splitext(fname)
        init_fname = os.path.join(path, fname + '_init.npz')
        fname = os.path.join(path, fname + '.npz')

        if os.path.exists(init_fname):
            init_param_values_file = np.load(init_fname)
            init_param_values = [init_param_values_file['arr_%d' % i] for i in range(len(init_param_values_file.files))]
        else:
            model = self.model_class()
            init_param_values = model.get_param_values()

        param_values = self.hyperparam_exp_get_model(init_param_values)

        np.savez(init_fname, *init_param_values)
        np.savez(fname, *param_values)
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('init_param_values_fname: "%s"\n' % init_fname)
            handle.write('param_values_fname: "%s"\n' % fname)
        return True



class MultipleChoiceTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(MultipleChoiceTest, self).__init__(question, testDict)
        self.ans = testDict['result']
        self.question = testDict['question']

    def execute(self, grades, moduleDict, solutionDict):
        studentSolution = str(getattr(moduleDict['answers'], self.question)())
        encryptedSolution = sha1(studentSolution.strip().lower()).hexdigest()
        if encryptedSolution == self.ans:
            return self.testPass(grades)
        else:
            self.addMessage("Solution is not correct.")
            self.addMessage("Student solution: %s" % studentSolution)
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# File intentionally blank.\n')
        handle.close()
        return True

