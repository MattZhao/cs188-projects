# layout.py
# ---------
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
from util import manhattanDistance
from game import Grid
# from capsule import DefaultCapsule
import os
import random
import itertools

VISIBILITY_MATRIX_CACHE = {}

PROB_FOOD_LEFT = 0.4
PROB_LEFT_TOP = 0.6
PROB_OPPOSITE_CORNERS = 0.7
PROB_FOOD_RED = 0.3
PROB_GHOST_RED = 0.6

PROB_BOTH_TOP = PROB_LEFT_TOP * (1 - PROB_OPPOSITE_CORNERS)
PROB_BOTH_BOTTOM = (1 - PROB_LEFT_TOP) * (1 - PROB_OPPOSITE_CORNERS)
PROB_ONLY_LEFT_TOP = PROB_LEFT_TOP * PROB_OPPOSITE_CORNERS
PROB_ONLY_LEFT_BOTTOM = (1 - PROB_LEFT_TOP) * PROB_OPPOSITE_CORNERS

class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, layoutText=None, seed=None, width=None, height=None, vpi=False):
        if layoutText:
            self.width = len(layoutText[0])
            self.height= len(layoutText)
            self.walls = Grid(self.width, self.height, False)
            self.redWalls = Grid(self.width, self.height, False)
            self.blueWalls = Grid(self.width, self.height, False)
            self.food = Grid(self.width, self.height, False)
            self.capsules = []
            self.agentPositions = []
            self.numGhosts = 0
            self.processLayoutText(layoutText)
            self.layoutText = layoutText
            self.totalFood = len(self.food.asList())
        elif vpi:
            layoutText = generateVPIHuntersBoard(seed)
            self.__init__(layoutText)
        else:
            layoutText = generateRandomHuntersBoard(seed, width, height)
            self.__init__(layoutText)

    def getNumGhosts(self):
        return self.numGhosts

    def initializeVisibilityMatrix(self):
        global VISIBILITY_MATRIX_CACHE
        if reduce(str.__add__, self.layoutText) not in VISIBILITY_MATRIX_CACHE:
            from game import Directions
            vecs = [(-0.5,0), (0.5,0),(0,-0.5),(0,0.5)]
            dirs = [Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
            vis = Grid(self.width, self.height, {Directions.NORTH:set(), Directions.SOUTH:set(), Directions.EAST:set(), Directions.WEST:set(), Directions.STOP:set()})
            for x in range(self.width):
                for y in range(self.height):
                    if self.walls[x][y] == False:
                        for vec, direction in zip(vecs, dirs):
                            dx, dy = vec
                            nextx, nexty = x + dx, y + dy
                            while (nextx + nexty) != int(nextx) + int(nexty) or not self.walls[int(nextx)][int(nexty)] :
                                vis[x][y][direction].add((nextx, nexty))
                                nextx, nexty = x + dx, y + dy
            self.visibility = vis
            VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)] = vis
        else:
            self.visibility = VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)]

    def isWall(self, pos):
        x, col = pos
        return self.walls[x][col]

    def getRandomLegalPosition(self):
        x = random.choice(range(self.width))
        y = random.choice(range(self.height))
        while self.isWall( (x, y) ):
            x = random.choice(range(self.width))
            y = random.choice(range(self.height))
        return (x,y)

    def getRandomCorner(self):
        poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        return random.choice(poses)

    def getFurthestCorner(self, pacPos):
        poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        dist, pos = max([(manhattanDistance(p, pacPos), p) for p in poses])
        return pos

    def isVisibleFrom(self, ghostPos, pacPos, pacDirection):
        row, col = [int(x) for x in pacPos]
        return ghostPos in self.visibility[row][col][pacDirection]

    def __str__(self):
        return "\n".join(self.layoutText)

    def deepCopy(self):
        return Layout(self.layoutText[:])

    def processLayoutText(self, layoutText):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
         % - Wall
         . - Food
         o - Capsule
         G - Ghost
         P - Pacman
         B - Blue Wall
         R - Red Wall
        Other characters are ignored.
        """
        maxY = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layoutChar = layoutText[maxY - y][x]
                self.processLayoutChar(x, y, layoutChar)
        self.agentPositions.sort()
        self.agentPositions = [ ( i == 0, pos) for i, pos in self.agentPositions]

    def processLayoutChar(self, x, y, layoutChar):
        if layoutChar == '%':
            self.walls[x][y] = True
        elif layoutChar == 'B':
            self.blueWalls[x][y] = True
        elif layoutChar == 'R':
            self.redWalls[x][y] = True
        elif layoutChar == '.':
            self.food[x][y] = True
        elif layoutChar == 'o':
            self.capsules.append(DefaultCapsule(x, y))
        elif layoutChar == 'P':
            self.agentPositions.append( (0, (x, y) ) )
        elif layoutChar in ['G']:
            self.agentPositions.append( (1, (x, y) ) )
            self.numGhosts += 1
        elif layoutChar in  ['1', '2', '3', '4']:
            self.agentPositions.append( (int(layoutChar), (x,y)))
            self.numGhosts += 1
            
def getLayout(name, back = 2):
    if name.endswith('.lay'):
        layout = tryToLoad('layouts/' + name)
        if layout == None: layout = tryToLoad(name)
    else:
        layout = tryToLoad('layouts/' + name + '.lay')
        if layout == None: layout = tryToLoad(name + '.lay')
    if layout == None and back >= 0:
        curdir = os.path.abspath('.')
        os.chdir('..')
        layout = getLayout(name, back -1)
        os.chdir(curdir)
    return layout

def tryToLoad(fullname):
    if(not os.path.exists(fullname)): return None
    f = open(fullname)
    try: return Layout([line.strip() for line in f])
    finally: f.close()

def generateVPIHuntersBoard(seed=None):
    width = 11
    height = 11
    foodHouseLeft = util.flipCoin(PROB_FOOD_LEFT)

    layoutTextGrid = [[' ' for _ in xrange(width)] for _ in xrange(height)]
    layoutTextGrid[0] = ['%' for _ in xrange(width)]
    layoutTextGrid[-1] = layoutTextGrid[0][:]
    for i in xrange(height):
        layoutTextGrid[i][0] = layoutTextGrid[i][-1] = '%'
    possibleLocations = pickPossibleLocations(width, height)
    # (foodX, foodY), (ghostX, ghostY) = tuple(random.sample(possibleLocations, 2))

    bottomLeft, topLeft, bottomRight, topRight = tuple(possibleLocations)

    foodX, foodY = topLeft
    ghostX, ghostY = topRight
    if not util.flipCoin(PROB_FOOD_LEFT):
        (foodX, foodY), (ghostX, ghostY) = (ghostX, ghostY), (foodX, foodY)

    layoutTextGrid[-foodY-1][foodX] = '.'
    layoutTextGrid[-ghostY-1][ghostX] = 'G'
    for foodWallX, foodWallY in buildHouseAroundCenter(foodX, foodY):
        if util.flipCoin(PROB_FOOD_RED):
            layoutTextGrid[-foodWallY-1][foodWallX] = 'R'
        else:
            layoutTextGrid[-foodWallY-1][foodWallX] = 'B'
    for ghostWallX, ghostWallY in buildHouseAroundCenter(ghostX, ghostY):
        if util.flipCoin(PROB_GHOST_RED):
            layoutTextGrid[-ghostWallY-1][ghostWallX] = 'R'
        else:
            layoutTextGrid[-ghostWallY-1][ghostWallX] = 'B'
    layoutTextGrid[5][5] = 'P'
    layoutTextRowList = [''.join(row) for row in layoutTextGrid]
    return layoutTextRowList

def generateRandomHuntersBoard(seed=None, width=None, height=None):
    """Note that this is constructing a string, so indexing is [-y-1][x] rather than [x][y]"""
    random.seed(seed)

    leftHouseTop  = util.flipCoin(PROB_LEFT_TOP)

    if not width or not height:
        width = random.randrange(11, 20, 4)
        height = random.randrange(11, 16, 4)
    layoutTextGrid = [[' ' for _ in xrange(width)] for _ in xrange(height)]
    layoutTextGrid[0] = ['%' for _ in xrange(width)]
    layoutTextGrid[-1] = layoutTextGrid[0][:]
    for i in xrange(height):
        layoutTextGrid[i][0] = layoutTextGrid[i][-1] = '%'
    possibleLocations = pickPossibleLocations(width, height)
    # (foodX, foodY), (ghostX, ghostY) = tuple(random.sample(possibleLocations, 2))

    bottomLeft, topLeft, bottomRight, topRight = tuple(possibleLocations)

    if leftHouseTop:
        foodX, foodY = topLeft
        ghostX, ghostY = bottomRight if util.flipCoin(PROB_OPPOSITE_CORNERS) else topRight
    else:
        foodX, foodY = bottomLeft
        ghostX, ghostY = topRight if util.flipCoin(PROB_OPPOSITE_CORNERS) else bottomRight
    if not util.flipCoin(PROB_FOOD_LEFT):
        (foodX, foodY), (ghostX, ghostY) = (ghostX, ghostY), (foodX, foodY)

    layoutTextGrid[-foodY-1][foodX] = '.'
    layoutTextGrid[-ghostY-1][ghostX] = 'G'
    for foodWallX, foodWallY in buildHouseAroundCenter(foodX, foodY):
        if util.flipCoin(PROB_FOOD_RED):
            layoutTextGrid[-foodWallY-1][foodWallX] = 'R'
        else:
            layoutTextGrid[-foodWallY-1][foodWallX] = 'B'
    for ghostWallX, ghostWallY in buildHouseAroundCenter(ghostX, ghostY):
        if util.flipCoin(PROB_GHOST_RED):
            layoutTextGrid[-ghostWallY-1][ghostWallX] = 'R'
        else:
            layoutTextGrid[-ghostWallY-1][ghostWallX] = 'B'
    layoutTextGrid[-2][1] = 'P'
    layoutTextRowList = [''.join(row) for row in layoutTextGrid]
    return layoutTextRowList

def pickPossibleLocations(width, height):
    # return list(itertools.product(range(3, width - 3, 4), range(3, height - 3, 4)))
    return [(3, 3), (3, height - 4), (width - 4, 3), (width - 4, height - 4)]

def buildHouseAroundCenter(x, y):
    return set(itertools.product([x-1, x, x+1], [y-1, y, y+1])) - {(x, y), (x, y-1)}

if __name__ == '__main__':
    lay = Layout(generateVPIHuntersBoard())
    print lay
