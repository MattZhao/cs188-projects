# features.py
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
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
        add an extra feature that shows the number of continuous white regions in the graph.
    ##
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    """===== setup for enhanced extraction ===="""
    m, n = datum.shape[0], datum.shape[1]
    num_w_regions = 0
    visited = set()
    i, j = newStartPoint(visited, features, m, n)

    from collections import deque
    queue = deque()
    while i >= 0 and j >= 0:
        queue.append((i, j))
        num_w_regions += 1

        while queue:
            x, y = queue.popleft()
            if y > 0 and features[x][y - 1] == 0 and (x, y - 1) not in visited:
                visited.add((x, y - 1))
                queue.append((x, y - 1))

            if y + 1 < n and features[x][y + 1] == 0 and (x, y + 1) not in visited:
                visited.add((x, y + 1))
                queue.append((x, y + 1))

            if x + 1 < m and features[x + 1][y] == 0 and (x + 1, y) not in visited:
                visited.add((x + 1, y))
                queue.append((x + 1, y))

            if x > 0 and features[x - 1][y] == 0 and (x - 1, y) not in visited:
                visited.add((x - 1, y))
                queue.append((x - 1, y))

        i, j = newStartPoint(visited, features, m, n)

    extra_features = np.array([0, 0, 0])
    if num_w_regions == 1:
        extra_features = np.array([1, 0, 0])
    elif num_w_regions == 2:
        extra_features = np.array([0, 1, 0])
    elif num_w_regions > 2:
        extra_features = np.array([0, 0, 1])
    return np.concatenate((features.flatten(), extra_features), axis = 0)

def newStartPoint(visited, features, m, n):
    for i in range(m):
        for j in range(n):
            if features[i][j] == 0 and (i, j) not in visited:
                visited.add((i, j))
                return i, j
    return -1, -1

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
