# pacmanPlot.py
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


from graphicsDisplay import PacmanGraphics
from graphicsDisplay import InfoPane
import graphicsDisplay
import graphicsUtils
from game import GameStateData
from game import AgentState
from game import Configuration
from game import Directions
from layout import Layout
from Tkinter import mainloop
import math
import numpy as np
import time
import plotUtil

LINE_COLOR = graphicsUtils.formatColor(0, 1, 0)

def plotPoints(x,y):
    """
    Create a Pacman display, plotting the points (x[i],y[i]) for all i in len(x).
    This method will block control and hand it to the displayed window.

    x: array or list of N scalar values.
    y: array or list of N scalar values.

    >>> x = range(-3,4)
    >>> squared = lambda x : x**2
    >>> y = map(squared, x)
    >>> pacmanPlot.plotPoints(x,y)
   """
    display = PacmanPlot(x,y)
    display.takeControl()

class PacmanPlot(PacmanGraphics):
    def __init__(self, x=None, y=None, zoom=1.0, frameTime=0.0):
        """
        Create and dispaly a pacman plot figure.

        If both x and y are provided, plot the points (x[i],y[i]) for all i in len(x).

        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.

        x: array or list of N scalar values. Default=None, in which case no points will be plotted
        y: array or list of N scalar values. Default=None, in which case no points will be plotted
        """
        super(PacmanPlot, self).__init__(zoom, frameTime)

        if x is None or y is None:
            width = 23
            height = 23
            xmin = -(width-1)/2+1
            ymin = -(height-1)/2+1

            self.initPlot(xmin, ymin, width, height)
        else:
            self.plot(x,y)

    def initPlot(self, xmin, ymin, width, height):
        if graphicsUtils._canvas is not None:
            graphicsUtils.clear_screen()

        # Initialize GameStateData with blank board with axes
        self.width = width
        self.height = height
        self.xShift = -(xmin-1)
        self.yShift = -(ymin-1)
        self.line = None

        self.zoom = min(30.0/self.width, 20.0/self.height)
        self.gridSize = graphicsDisplay.DEFAULT_GRID_SIZE * self.zoom


#         fullRow = ['%']*self.width
#         row = ((self.width-1)/2)*[' '] + ['%'] + ((self.width-1)/2)*[' ']
#         boardText = ((self.height-1)/2)*[row] + [fullRow] + ((self.height-1)/2)*[row]

        numSpaces = self.width-1
        numSpacesLeft = self.xShift
        numSpacesRight = numSpaces-numSpacesLeft

        numRows = self.height
        numRowsBelow = self.yShift
        numRowsAbove = numRows-1-numRowsBelow


        fullRow = ['%']*self.width
        if numSpacesLeft < 0:
            row = [' ']*self.width
        else:
            row = numSpacesLeft*[' '] + ['%'] + numSpacesRight*[' ']
        boardText = numRowsAbove*[row] + [fullRow] + numRowsBelow*[row]

        layout = Layout(boardText)

        self.blankGameState = GameStateData()
        self.blankGameState.initialize(layout, 0)
        self.initialize(self.blankGameState)
        title = 'Pacman Plot'
        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()

    def plot(self, x, y, weights=None, title='Pacman Plot'):
        """
        Plot the input values x with their corresponding output values y (either true or predicted).
        Also, plot the linear regression line if weights are given; assuming h_w(x) = weights[0]*x + weights[1].

        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.

        x: array or list of N scalar values.
        y: array or list of N scalar values.
        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return

        if isinstance(x[0], np.ndarray):
            # Scrape the first element of each data point
            x = [data[0] for data in x]

        xmin = int(math.floor(min(x)))
        ymin = int(math.floor(min(y)))
        xmax = int(math.ceil(max(x)))
        ymax = int(math.ceil(max(y)))
        width = xmax-xmin+3
        height = ymax-ymin+3
        self.initPlot(xmin, ymin, width, height)

        gameState = self.blankGameState.deepCopy()

        gameState.agentStates = []

        # Add ghost at each point
        for (px,py) in zip(x,y):
            point = (px+self.xShift, py+self.yShift)
            gameState.agentStates.append( AgentState( Configuration( point, Directions.STOP), False) )

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()


    def setWeights(self, weights):
        pass

    def takeControl(self):
        """
        Give full control to the window. Blocks current thread. Program will exit when window is closed.
        """
        mainloop()

#     def animate():
#         numSteps = width-2
#         for i in range(numSteps):
#             x = point1[0] + i*dx*1.0/numSteps
#             y = point1[1] + i*dy*1.0/numSteps
#             newPacman = AgentState( Configuration( (x,y), angle), True)
#             display.animatePacman(newPacman, display.agentImages[0][0], display.agentImages[0][1])
#             display.agentImages[0] = (newPacman, display.agentImages[0][1])
#         newPacman = AgentState( Configuration( point2, angle), True)
#         display.animatePacman(newPacman, display.agentImages[0][0], display.agentImages[0][1])
#         display.agentImages[0] = (newPacman, display.agentImages[0][1])


class PacmanPlotRegression(PacmanPlot):
    def __init__(self, zoom=1.0, frameTime=0.0):
        super(PacmanPlotRegression, self).__init__(zoom=zoom, frameTime=frameTime)
        self.addPacmanToLineStart = True

    def plot(self, x, y, weights=None, title='Linear Regression'):
        """
        Plot the input values x with their corresponding output values y (either true or predicted).
        Also, plot the linear regression line if weights are given; assuming h_w(x) = weights[0]*x + weights[1].

        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.

        x: array or list of N scalar values.
        y: array or list of N scalar values.
        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return

        if isinstance(x[0], np.ndarray):
            # Scrape the first element of each data point
            x = [data[0] for data in x]

        xmin = int(math.floor(min(x)))
        ymin = int(math.floor(min(y)))
        xmax = int(math.ceil(max(x)))
        ymax = int(math.ceil(max(y)))
        width = xmax-xmin+3
        height = ymax-ymin+3
        self.initPlot(xmin, ymin, width, height)

        gameState = self.blankGameState.deepCopy()

        gameState.agentStates = []

        # Put pacman in bottom left
        if self.addPacmanToLineStart is True:
            gameState.agentStates.append( AgentState( Configuration( (1,1), Directions.STOP), True) )

        # Add ghost at each point
        for (px,py) in zip(x,y):
            point = (px+self.xShift, py+self.yShift)
            gameState.agentStates.append( AgentState( Configuration( point, Directions.STOP), False) )

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()

        if weights is not None:
            self.setWeights(weights)

    def setWeights(self, weights):
        """
        Plot the linear regression line for given weights; assuming h_w(x) = weights[0]*x + weights[1].

        This will draw on the existing pacman window with the existing points

        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        weights = np.array(weights)

        if weights.size >= 2:
            w = weights[0]
            b = weights[1]
        else:
            w = float(weights)
            b = 0

#         xmin = min(x)
#         xmax = max(x)
#
#         ymin = w*xmin + b
#         ymax = w*xmax + b
#
#         point1 = (xmin+self.xShift, ymin+self.yShift)
#         point2 = (xmax+self.xShift, ymax+self.yShift)

        (point1, point2) = plotUtil.lineBoxIntersection(w, -1, b,
                                                        1-self.xShift, 1-self.yShift,
                                                        self.width-2-self.xShift, self.height-2-self.yShift)

        if point1 is not None and point2 is not None:
            point1 = (point1[0]+self.xShift, point1[1]+self.yShift)
            point2 = (point2[0]+self.xShift, point2[1]+self.yShift)

            dx = point2[0]-point1[0]
            dy = point2[1]-point1[1]
            if dx == 0:
                angle = 90 + 180*dy*1.0/abs(dy)
            else:
                angle = math.atan(dy*1.0/dx)*180.0/math.pi

            if self.line is not None:
                graphicsUtils.remove_from_screen(self.line)
            self.line = graphicsUtils.polygon([self.to_screen(point1), self.to_screen(point2)], LINE_COLOR, filled=0, behind=0)

            if self.addPacmanToLineStart is True and len(self.agentImages) > 0:
                # Bring pacman to front of display
                graphicsUtils._canvas.tag_raise(self.agentImages[0][1][0])

                # Put pacman at beginning of line
                self.movePacman(point1, angle, self.agentImages[0][1])

            graphicsUtils.refresh()

class PacmanPlotLogisticRegression1D(PacmanPlot):
    def __init__(self, zoom=1.0, frameTime=0.0):
        super(PacmanPlotLogisticRegression1D, self).__init__(zoom=zoom, frameTime=frameTime)
        self.addPacmanToLineStart = False

    def plot(self, x, y, weights=None, title='Logistic Regression'):
        """
        Plot the 1D input points, data[i], colored based on their corresponding labels (either true or predicted).
        Also, plot the logistic function fit if weights are given.

        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.

        x: list of 1D points, where each 1D point in the list is a 1 element numpy.ndarray
        y: list of N labels, one for each point in data. Labels can be of any type that can be converted
            a string.
        weights: array of 2 values the first one is the weight on the data and the second value is the bias weight term.
        If there are only 1 values in weights,
            the bias term is assumed to be zero.  If None, no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return

        # Process data, sorting by label
        possibleLabels = list(set(y))
        sortedX = {}
        for label in possibleLabels:
            sortedX[label] = []

        for i in range(len(x)):
            sortedX[y[i]].append(x[i])

        xmin = int(math.floor(min(x)))
        xmax = int(math.ceil(max(x)))
        ymin = int(math.floor(0))-1
        ymax = int(math.ceil(1))
        width = xmax-xmin+3
        height = ymax-ymin+3
        self.initPlot(xmin, ymin, width, height)

        gameState = self.blankGameState.deepCopy()

        gameState.agentStates = []

        # Put pacman in bottom left
        if self.addPacmanToLineStart is True:
            gameState.agentStates.append( AgentState( Configuration( (1,1), Directions.STOP), True) )

        # Add ghost at each point
        for (py, label) in enumerate(possibleLabels):
            pointsX = sortedX[label]
            for px in pointsX:
                point = (px+self.xShift, py+self.yShift)
                agent = AgentState( Configuration( point, Directions.STOP), False)
                agent.isPacman = 1-py
                gameState.agentStates.append(agent)

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()

        if weights is not None:
            self.setWeights(weights)

    def setWeights(self, weights):
        """
        Plot the logistic regression line for given weights

        This will draw on the existing pacman window with the existing points

        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """
        weights = np.array(weights)

        if weights.size >= 2:
            w = weights[0]
            b = weights[1]
        else:
            w = float(weights)
            b = 0

        xmin = 1 - self.xShift
        xmax = self.width-2 - self.xShift

        x = np.linspace(xmin, xmax,30)
        y = 1.0/(1+np.exp(-(w*x+b)))
        x += self.xShift
        y += self.yShift

        if self.line is not None:
            for obj in self.line:
                graphicsUtils.remove_from_screen(obj)

        self.line = []

        prevPoint = self.to_screen((x[0],y[0]))
        for i in xrange(1,len(x)):
            point = self.to_screen((x[i],y[i]))
            self.line.append(graphicsUtils.line(prevPoint, point, LINE_COLOR))
            prevPoint = point

#         prevPoint = self.to_screen((x[0],y[0]))
#         for i in xrange(1,len(x)):
#             point = self.to_screen((x[i],y[i]))
#             line.append(graphicsUtils.line(prevPoint, point, LINE_COLOR, filled=0, behind=0)
#
#             prevPoint = point


        if self.addPacmanToLineStart is True and len(self.agentImages) > 0:
            # Bring pacman to front of display
            graphicsUtils._canvas.tag_raise(self.agentImages[0][1][0])

            # Put pacman at beginning of line
            if w >= 0:
                self.movePacman((x[0]-0.5,y[0]), Directions.EAST, self.agentImages[0][1])
            else:
                self.movePacman((x[-1]+0.5,y[-1]), Directions.WEST, self.agentImages[0][1])

        graphicsUtils.refresh()

class PacmanPlotClassification2D(PacmanPlot):
    def __init__(self, zoom=1.0, frameTime=0.0):
        super(PacmanPlotClassification2D, self).__init__(zoom=zoom, frameTime=frameTime)
        self.prev_weights = None

    def plot(self, x, y, weights=None, title='Linear Classification'):
        """
        Plot the 2D input points, data[i], colored based on their corresponding labels (either true or predicted).
        Also, plot the linear separator line if weights are given.

        This will draw on the existing pacman window (clearing it first) or create a new one if no window exists.

        x: list of 2D points, where each 2D point in the list is a 2 element numpy.ndarray
        y: list of N labels, one for each point in data. Labels can be of any type that can be converted
            a string.
        weights: array of 3 values the first two are the weight on the data and the third value is the bias
        weight term. If there are only 2 values in weights, the bias term is assumed to be zero.  If None,
        no line is drawn. Default: None
        """
        if np.array(x).size == 0:
            return

        # Process data, sorting by label
        possibleLabels = list(set(y))
        sortedX1 = {}
        sortedX2 = {}
        for label in possibleLabels:
            sortedX1[label] = []
            sortedX2[label] = []

        for i in range(len(x)):
            sortedX1[y[i]].append(x[i][0])
            sortedX2[y[i]].append(x[i][1])

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

        x1min = int(math.floor(x1min))
        x1max = int(math.ceil(x1max))
        x2min = int(math.floor(x2min))
        x2max = int(math.ceil(x2max))

        width = x1max-x1min+3
        height = x2max-x2min+3
        self.initPlot(x1min, x2min, width, height)

        gameState = self.blankGameState.deepCopy()

        gameState.agentStates = []

        # Add ghost/pacman at each point
        for (labelIndex, label) in enumerate(possibleLabels):
            pointsX1 = sortedX1[label]
            pointsX2 = sortedX2[label]
            for (px, py) in zip(pointsX1, pointsX2):
                point = (px+self.xShift, py+self.yShift)
                agent = AgentState( Configuration( point, Directions.STOP), False)
                agent.isPacman = (labelIndex==0)
                if labelIndex==2:
                    agent.scaredTimer = 1
                gameState.agentStates.append(agent)

#         self.initialize(gameState)
        graphicsUtils.clear_screen()
        self.infoPane = InfoPane(gameState.layout, self.gridSize)
        self.drawStaticObjects(gameState)
        self.drawAgentObjects(gameState)

        graphicsUtils.changeText(self.infoPane.scoreText, title)
        graphicsUtils.refresh()

        if weights is not None:
            self.setWeights(weights)

    def setWeights(self, weights):
        """
        Plot the logistic regression line for given weights

        This will draw on the existing pacman window with the existing points

        weights: array or list of 2 values (or if just one value, the bias weight is assumed to be zero). If None,
            no line is drawn. Default: None
        """

        weights = np.array(weights)

        if self.prev_weights is not None:
            if np.allclose(self.prev_weights, weights):
                return False # no line update
        self.prev_weights = weights.copy()

        w1 = weights[0]
        w2 = weights[1]
        if weights.size >= 3:
            b = weights[2]
        else:
            b = 0

        # Line functions
        # Line where w1*x1 + w2*x2 + b = 0
        # x2 = -(w1*x1 + b)/w2  or
        # x1 = -(w2*x2 + b)/w1

        # Figure out where line intersections bounding box around points
        if w1 == 0 and w2 == 0:
            return

        (point1, point2) = plotUtil.lineBoxIntersection(w1, w2, b,
                                                        1-self.xShift, 1-self.yShift,
                                                        self.width-2-self.xShift, self.height-2-self.yShift)

        if point1 is not None and point2 is not None:
            point1 = (point1[0]+self.xShift, point1[1]+self.yShift)
            point2 = (point2[0]+self.xShift, point2[1]+self.yShift)

            if self.line is not None:
                graphicsUtils.remove_from_screen(self.line)
            self.line = graphicsUtils.polygon([self.to_screen(point1), self.to_screen(point2)], LINE_COLOR, filled=0, behind=0)

        graphicsUtils.refresh()
        return True # updated line

if __name__ == '__main__':
    """
    Demo code
    """
    # Regression

    display = PacmanPlotRegression()

    x = np.random.normal(0,1,10)
    display.plot(x*3, x**3)
    time.sleep(2)

    for i in range(6):
        display.setWeights([(5-i)/5.0])
        time.sleep(1)
    time.sleep(1)

    # With offset
    display.setWeights([0, -5])
    time.sleep(2)

    # Classification 2D

    display = PacmanPlotClassification2D()

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

    display.plot(data, labels)
    time.sleep(2)

    for i in range(8):
        display.setWeights([4,i])
        time.sleep(1)


    # With offset and horizontal separator
    display.setWeights([0, 1, -3])
    time.sleep(2)

    # Logistic Regression

    display = PacmanPlotLogisticRegression1D()

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

    display.plot(data, labels)
    time.sleep(2)

    for i in range(8):
        display.setWeights(4-i)
        time.sleep(1)
    time.sleep(1)

    # With offset and horizontal separator
    display.setWeights([-3, -6])

#     # Just some extra tests
#     display = PacmanPlotRegression()
#     display.plot([-2, 2, 2], [2, 2, -2], [4,0])
#     display = PacmanPlotClassification2D()
#     display.plot([np.array([-1,1]), np.ones(2), np.array([1,-1])], [0,1,2], [1,0])

    display.takeControl()
    # Blocked until the window is closed
