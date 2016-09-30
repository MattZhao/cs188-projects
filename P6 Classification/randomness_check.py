# randomness_check.py
# -------------------
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
import models
import solvers
import tensorflow_util as tfu


def tinyDataSet():
    training = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1], [1,1,1]])
    trainingLabels = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,0], [1,0]])

    validation = np.array([[1,0,1]])
    validationLabels = np.array([[0,1]])

    test = np.array([[1,0,1]])
    testLabels = np.array([[1,0]])

    return (training, trainingLabels, validation, validationLabels, test, testLabels)


def main():
    dataset = tinyDataset()
    input_data, target_data = dataset[:2]
    W_init = np.array([[0.4, 0.0], [0.0, -0.2], [0.1, 0.0]], dtype=np.float32)
    b_init = np.array([-0.5, 0.3], dtype=np.float32)
    model = models.LinearRegressionModel(x_shape=(None, 3), W=W_init, b=b_init)
    solver = solvers.GradientDescentSolver(learning_rate=0.1, iterations=1, momentum=0.9)

    target_ph = tf.placeholder(tf.float32, shape=(None, 2))
    loss_tensor = solvers.squared_error(model.prediction_tensor, target_ph)
    param_vars = model.get_param_vars(trainable=True)

    updates = solver.get_updates(loss_tensor, param_vars)
    update_ops = [tf.assign(old, new) for (old, new) in updates]

    # gradient and parameter values before updates
    grad_tensors = tf.gradients(loss_tensor, param_vars)
    feed_dict = dict([(model.input_ph, input_data), (target_ph, target_data)])
    grads = [grad_tensor.eval(session=tfu.get_session(), feed_dict=feed_dict) for grad_tensor in grad_tensors]
    param_values = model.get_param_values()

    print(grads)
    print(param_values)
    tfu.get_session().run([loss_tensor] + update_ops, feed_dict=feed_dict)
    print(model.get_param_values())

if __name__ == "__main__":
    main()

