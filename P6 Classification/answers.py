# answers.py
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


import util


def q2():
    "*** YOUR CODE HERE ***"
    return 'b'

def q3():
    "*** YOUR CODE HERE ***"
    return 'a'

def q7():
    "*** YOUR CODE HERE ***"
    return 'both'

def q10():
    """
    Returns a dict of hyperparameters.

    Returns:
        A dict with the learning rate and momentum.

    You should find the hyperparameters by empirically finding the values that
    give you the best validation accuracy when the model is optimized for 1000
    iterations. You should achieve at least a 97% accuracy on the MNIST test set.
    """
    hyperparams = dict()
    hyperparams['learning_rate'] = 0.01
    hyperparams['momentum'] = 0.9
    "*** YOUR CODE HERE ***"
    # filter out any item in the dict that is not the learning rate nor momentum
    allowed_hyperparams = ['learning_rate', 'momentum']
    hyperparams = dict([(k, v) for (k, v) in hyperparams.items() if k in allowed_hyperparams])
    return hyperparams
