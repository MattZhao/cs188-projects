# plotUtil.py
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


'''
Utility module to help with plotting regression and classification data.
'''
import numpy as np
try:
    import matplotlib
    # Included to avoid crashing python when matplot lib and some other
    # Tkinter app (e.g. pacman) are both involved
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    matplotlibMissing = False
except ImportError:
    print 'plotUtil.plotRegression: Sorry, could not import matplotlib'
    matplotlibMissing = True


def plotCurve(x, y, figureIdx=1, figureTitle='Curve',blocking=False):
    """
    To plot a simple curve
    """
    if matplotlibMissing:
        return
    plt.figure(figureIdx)
    plt.figure(figureIdx).clf()
    plt.plot(x, y,'-r')
    plt.title(figureTitle)
    plt.pause(0.01)

    if blocking is True:
        plt.show()

def plotTwoCurves(x1, y1, x2, y2, figureIdx=1, figureTitle='Curve', blocking=False, showLegend='False', label1='label1', label2='label2'):
    """
    To plot a simple curve
    """
    if matplotlibMissing:
        return
    plt.figure(figureIdx)
    plt.figure(figureIdx).clf()
    plt.plot(x1, y1,'-r',label=label1)
    plt.plot(x2, y2,'-b',label=label2)

    plt.title(figureTitle)
    if showLegend is True:
        plt.legend(numpoints=1)
    plt.pause(0.01)

    if blocking is True:
        plt.show()

def plotRegression(x, y, weights=None, figureIdx=1, blocking=False, showLegend='False', figureTitle = 'Linear Regression'):
    """
    Plot the input values x with their corresponding output values y (either true or predicted).
    Also, plot the linear regression line if weights are given; assuming h_w(x) = weights[0]*x + weights[1].

    This will draw on any current matplotlib figure (clearing it first) or create a new one if no figure exists.

    x: array or list of N scalar values.
    y: array or list of N scalar values.
    weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
        no line is drawn. Default: None

    blocking: If true, the function will not return until the figure has been closed. Default: False
    """
    if matplotlibMissing:
        return

    plt.figure(figureIdx)
    plt.figure(figureIdx).clf()
    
    if np.array(x).size == 0:
        return
    
    if isinstance(x[0], np.ndarray):
        # Scrape the first element of each data point
        x = [data[0] for data in x]

    plt.plot(x, y, 'bo', label='(x[i], y[i])')

    if weights is not None:
        weights = np.array(weights)

        if weights.size >= 2:
            w = weights[0]
            b = weights[1]
        else:
            w = weights
            b = 0

        xmin = min(x)
        xmax = max(x)

        ymin = w*xmin + b
        ymax = w*xmax + b

        plt.plot([xmin, xmax], [ymin, ymax], 'r-', label='y = w[1]*x + w[2]')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(figureTitle)
    if showLegend is True:
        plt.legend(numpoints=1)

    plt.pause(0.01)
    if blocking is True:
        plt.show()

def plotLogisticRegression1D(data, labels, weights=None, blocking=False, showLegend='False', figureTitle = "Logistic Regression"):
    """
    Plot the 1D input points, data[i], colored based on their corresponding labels (either true or predicted).
    Also, plot the logistic function fit if weights are given.

    This will draw on any current matplotlib figure (clearing it first) or create a new one if no figure exists.

    data: list of 1D points, where each 1D point in the list is a 1 element numpy.ndarray
    labels: list of N labels, one for each point in data. Labels can be of any type that can be converted
        a string.
    weights: array of 2 values the first one is the weight on the data and the second value is the bias weight term.
    If there are only 1 values in weights,
        the bias term is assumed to be zero.  If None, no line is drawn. Default: None

    blocking: If true, the function will not return until the figure has been closed. Default: False
    """
    if matplotlibMissing:
        return

    # Process data, sorting by label
    possibleLabels = list(set(labels))
    sortedX = {}
    for label in possibleLabels:
        sortedX[label] = []

    for i in range(len(data)):
        sortedX[labels[i]].append(data[i])

    xmin = min(data)
    xmax = max(data)

    plt.clf()

    markers = ['o', 'o']
    colors = ['r', 'g']

    for (i, label) in enumerate(possibleLabels):
        marker = colors[i%len(colors)] + markers[i%len(markers)]
        plt.plot(sortedX[label], [i]*len(sortedX[label]), marker, label=str(label))
        plt.axis([xmin, xmax, -0.5, 1.5])

    if weights is not None:
        weights = np.array(weights)

        if weights.shape == ():
            w = weights
        else:
            w = weights[0]
        if weights.size >= 2:
            b = weights[1]
        else:
            b = 0

        x = np.linspace(xmin, xmax,100)
        y = 1.0/(1+np.exp(-(w*x+b)))

        plt.plot(x, y, 'b-', label='Logistic fit')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Logistic Regression')
    if showLegend is True:
        plt.legend(numpoints=1)

    plt.pause(0.01)
    if blocking is True:
        plt.show()

def plotClassification2D(data, labels, weights=None, blocking=False, showLegend='False', figureTitle = "Linear Classification"):
    """
    Plot the 2D input points, data[i], colored based on their corresponding labels (either true or predicted).
    Also, plot the linear seperator line if weights are given; the line where 0 = w[0]*x1 + w[1]*x2 + w[2].

    This will draw on any current matplotlib figure (clearing it first) or create a new one if no figure exists.

    data: list of 2D points, where each 2D point in the list is a 2 element numpy.ndarray
    labels: list of N labels, one for each point in data. Labels can be of any type that can be converted
        a string.
    weights: array of 3 values the first two are the weights on the first and second coordinate, respectively,
        of the 2D data. The third value is the bias weight term. If there are only 2 values in weights,
        the bias term is assumed to be zero.  If None, no line is drawn. Default: None

    blocking: If true, the function will not return until the figure has been closed. Default: False
    """
    if matplotlibMissing:
        return

    # Process data, sorting by label
    possibleLabels = list(set(labels))
    sortedX1 = {}
    sortedX2 = {}
    for label in possibleLabels:
        sortedX1[label] = []
        sortedX2[label] = []

    for i in range(len(data)):
        sortedX1[labels[i]].append(data[i][0])
        sortedX2[labels[i]].append(data[i][1])

    plt.clf()

    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for (i, label) in enumerate(possibleLabels):
        marker = colors[i%len(colors)] + markers[i%len(markers)]
        plt.plot(sortedX1[label], sortedX2[label], marker, label=label)

    if weights is not None:
        weights = np.array(weights)

        w1 = weights[0]
        w2 = weights[1]
        if weights.size >= 3:
            b = weights[2]
        else:
            b = 0

        x1min = float("inf")
        x1max = float("-inf")
        for x1Values in sortedX1.values():
            x1min = min(min(x1Values), x1min)
            x1max = max(max(x1Values), x1max)
        x2min = float("inf")
        x2max = float("-inf")
        for x2Values in sortedX2.values():
            x2min = min(min(x2Values), x2min)
            x2max = max(max(x2Values), x2max)

        # Line functions
        # Line where w1*x1 + w2*x2 + b = 0
        # x2 = -(w1*x1 + b)/w2  or
        # x1 = -(w2*x2 + b)/w1

        # Figure out where line intersections bounding box around points

        (point1, point2) = lineBoxIntersection(w1, w2, b, x1min, x2min, x1max, x2max)

        if point1 is not None and point2 is not None:
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', label='0 = w[0]*x1 + w[1]*x2 + w[2]')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Linear Classification')
    if showLegend is True:
        plt.legend(numpoints=1)

    plt.pause(0.01)
    if blocking is True:
        plt.show()

def lineBoxIntersection(w1, w2, b, xmin, ymin, xmax, ymax):
    """
    Figure out where line (w1*x+w2*y+b=0) intersections the 
    box (xmin, ymin) -> (xmax, ymax)
    """
        
    point1 = None
    point2 = None
    if w2 == 0:
        x1a = -(w2*ymin + b)*1.0/w1
        x1b = -(w2*ymax + b)*1.0/w1
    
        point1 = (x1a, ymin)
        point2 = (x1b, ymax)
    else:
        x2a = -(w1*xmin + b)*1.0/w2
        x2b = -(w1*xmax + b)*1.0/w2
        
        if w1 == 0:
            point1 = (xmin, x2a)
            point2 = (xmax, x2b)
        else:

            x1a = -(w2*ymin + b)*1.0/w1
            x1b = -(w2*ymax + b)*1.0/w1
            # Point 1
            if x2a < ymin:
                if xmin <= x1a and x1a <= xmax:
                    # Point 1 on bottom edge
                    point1 = (x1a, ymin)
            elif x2a > ymax:
                if xmin <= x1b and x1b <= xmax:
                    # Point 1 on top edge
                    point1 = (x1b, ymax)
            else:
                # Point 1 on left edge
                point1 = (xmin, x2a)
                
            # Point 2
            if point1 is not None:
                if x2b < ymin:
                    # Point 2 on bottom edge
                    point2 = (x1a, ymin)
                elif x2b > ymax:
                    # Point 2 on top edge
                    point2 = (x1b, ymax)
                else:
                    # Point 2 on right edge
                    point2 = (xmax, x2b)                                                
    return (point1, point2)


if __name__ == '__main__':
    """
    Demo code
    """
    # Regression
 
    x = np.random.normal(0,1,10)
    plotRegression(x, x**3, blocking=True)
    # Blocked until the window is closed
 
    for i in range(8):
        plotRegression(x, x**3, [i])
 
    # With offset
    plotRegression(x, x**3, [8, -10], blocking=True)
    # Blocked until the window is closed
 
    # Classification 2D
 
    # Generate labeled points
    means = ((4,4), (-4,4), (0,-4))
    labelNames= ('A','B','C')
    labels = []
    data = []
    for i in range(15):
        labelIndex = np.random.randint(len(labelNames))
        labels.append(labelNames[labelIndex])
        mean = np.array(means[labelIndex])
        data.append(np.random.normal(mean,1,mean.shape))
 
    plotClassification2D(data, labels, blocking=True)
    # Blocked until the window is closed
 
    for i in range(8):
        plotClassification2D(data, labels, [4,i])
 
 
    # With offset and horizontal separator
    plotClassification2D(data, labels, [0, 1, -3], blocking=True)
    # Blocked until the window is closed
    
    # Classification 1D

    # Generate labeled points
    means = (1, -5)
    labelNames= ('A','B')
    labels = []
    data = []
    for i in range(15):
        labelIndex = np.random.randint(len(labelNames))
        labels.append(labelNames[labelIndex])
        mean = np.array(means[labelIndex])
        data.append(np.random.normal(mean,3,mean.shape))

    plotLogisticRegression1D(data, labels, blocking=True)
    # Blocked until the window is closed

    for i in range(8):
        plotLogisticRegression1D(data, labels, 4-i)


    # With offset and horizontal separator
    plotLogisticRegression1D(data, labels, [-3, -6], blocking=True)
    # Blocked until the window is closed
    
    


