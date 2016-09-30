# datasets.py
# -----------
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
from tensorflow.examples.tutorials.mnist import input_data
from util import normal

_MNIST = None
def datasetA():
    training = np.array([[-3,3],[-3,0],[-1,1],[1,5],[3,4],[1,-1],[3,2],[5,0]])
    trainingLabels = np.ones((8,2))
    trainingLabels[:4,0] = 0
    trainingLabels[4:,1] = 0

    validation = np.array([[3,0]])
    validationLabels = np.array([[1,0]])

    test = np.array([[-1,2]])
    testLabels = np.array([[0,1]])
    return [training,trainingLabels, validation, validationLabels, test, testLabels]

def datasetB():
    X_1 = normal(shape=(100, 2), mean=1.5)*3
    X_2 = normal(shape=(100, 2), mean=-1)*3
    training = np.r_[X_1[:50], X_2[:50]]
    trainingLabels = np.ones((100,2))
    trainingLabels[50:,0] = 0
    trainingLabels[:50,1] = 0

    validation = np.r_[X_1[50:80], X_2[50:80]]
    validationLabels = np.ones((60,2))
    validationLabels[30:,0] = 0
    validationLabels[:30,1] = 0

    test = np.r_[X_1[80:], X_2[80:]]
    testLabels = np.ones((40,2))
    testLabels[20:, 0] = 0
    testLabels[:20, 1] = 0
    return [training, trainingLabels, validation, validationLabels, test, testLabels]

def tinyDataset():
    training = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1], [1,1,1]])
    trainingLabels = np.array([[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,0], [1,0]])

    validation = np.array([[1,0,1]])
    validationLabels = np.array([[0,1]])

    test = np.array([[1,0,1]])
    testLabels = np.array([[1,0]])

    return [training, trainingLabels, validation, validationLabels, test, testLabels]

def mnistDataset():
    global _MNIST

    if _MNIST is None:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        _MNIST = (mnist.train.images, mnist.train.labels,
            mnist.validation.images, mnist.validation.labels,
            mnist.test.images, mnist.test.labels)

    return [data.copy() for data in _MNIST]

def largeMnistDataset():
    all_data = mnistDataset()
    training = all_data[0][:2000]
    trainingLabels = all_data[1][:2000]
    validation = all_data[2][:1000]
    validationLabels = all_data[3][:1000]
    test = all_data[4][:1000]
    testLabels = all_data[5][:1000]

    return [training, trainingLabels, validation, validationLabels, test, testLabels]

def medMnistDataset():
    all_data = mnistDataset()
    training = all_data[0][:1000]
    trainingLabels = all_data[1][:1000]
    validation = all_data[2][:200]
    validationLabels = all_data[3][:200]
    test = all_data[4][:200]
    testLabels = all_data[5][:200]

    return [training, trainingLabels, validation, validationLabels, test, testLabels]

def tinyMnistDataset():
    all_data = mnistDataset()
    return [data[:10] for data in all_data]
