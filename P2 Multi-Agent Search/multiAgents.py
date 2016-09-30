# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        foodsPos = newFood.asList()
        foodsPos = sorted(foodsPos, key = lambda pos: manhattanDistance(newPos, pos))
        closestFoodDist = 0
        if len(foodsPos) > 0:
          closestFoodDist = manhattanDistance(foodsPos[0], newPos)

        foodCount = successorGameState.getNumFood()
        foodFeature = - closestFoodDist - 15*foodCount

        activeGhostsPos = []
        for ghost in newGhostStates:
          if ghost.scaredTimer == 0:
            activeGhostsPos.append(ghost.getPosition())
        activeGhostsPos = sorted(activeGhostsPos, key = lambda pos: manhattanDistance(pos, newPos))
        closestActiveGhostDist = 0
        if len(activeGhostsPos) > 0:
          closestActiveGhostDist = manhattanDistance(activeGhostsPos[0], newPos)
        return closestActiveGhostDist + foodFeature

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts)

    def maximize(self, gameState, depth, numGhosts):
        """
          maximizing agent in minimax
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        maxVal = float("-inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
          successor = gameState.generateSuccessor(0, action)
          tempVal = self.minimize(successor, depth, 1, numGhosts)
          if tempVal > maxVal:
            maxVal = tempVal
            best_action = action

        # terminal max returns actions whereas intermediate max returns values
        if depth > 1:
          return maxVal
        return best_action

    def minimize(self, gameState, depth, agentIndex, numGhosts):
        """
          minimizing agent in minimax 
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        minVal = float("inf")
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        if agentIndex == numGhosts:
          if depth < self.depth:
            for successor in successors:
              minVal = min(minVal, self.maximize(successor, depth + 1, numGhosts))
          else:
            for successor in successors:
              minVal = min(minVal, self.evaluationFunction(successor))
        else:
          for successor in successors:
            minVal = min(minVal, self.minimize(successor, depth, agentIndex + 1, numGhosts))
        return minVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def maximize(self, gameState, depth, numGhosts, alpha, beta):
        """
          maximizing agent with alpha-beta pruning
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        maxVal = float("-inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
          successor = gameState.generateSuccessor(0, action)
          tempVal = self.minimize(successor, depth, 1, numGhosts, alpha, beta)
          if tempVal > maxVal:
            maxVal = tempVal
            best_action = action

          # prune 
          if maxVal > beta:
            return maxVal
          alpha = max(alpha, maxVal)

        # terminal max returns actions whereas intermediate max returns values
        if depth > 1:
          return maxVal
        return best_action


    def minimize(self, gameState, depth, agentIndex, numGhosts, alpha, beta):
        """
          minimizing agent with alpha-beta pruning
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        minVal = float("inf")
        for action in gameState.getLegalActions(agentIndex):
          successor = gameState.generateSuccessor(agentIndex, action)
          if agentIndex == numGhosts:
            if depth < self.depth:
              tempVal = self.maximize(successor, depth + 1, numGhosts, alpha, beta)
            else:
              tempVal = self.evaluationFunction(successor)
          else:
            tempVal = self.minimize(successor, depth, agentIndex + 1, numGhosts, alpha, beta)
          if tempVal < minVal:
            minVal = tempVal

          # prune
          if minVal < alpha:
            return minVal
          beta = min(beta, minVal)
        return minVal

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts)

    def maximize(self, gameState, depth, numGhosts):
        """
          maximizing agent in expectimax
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        maxVal = float("-inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
          successor = gameState.generateSuccessor(0, action)
          tempVal = self.getExpectedValue(successor, depth, 1, numGhosts)
          if tempVal > maxVal:
            maxVal = tempVal
            best_action = action

        # terminal max returns actions whereas intermediate max returns values
        if depth > 1:
          return maxVal
        return best_action

    def getExpectedValue(self, gameState, depth, agentIndex, numGhosts):
        """
          minimizing agent in minimax 
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        
        expectedValue = 0
        successor_prob = 1.0 / len(legalActions)
        if agentIndex == numGhosts:
          if depth < self.depth:
            for successor in successors:
              expectedValue += successor_prob * self.maximize(successor, depth + 1, numGhosts)
          else:
            for successor in successors:
              expectedValue += successor_prob * self.evaluationFunction(successor)
        else:
          for successor in successors:
            expectedValue += successor_prob * self.getExpectedValue(successor, depth, agentIndex + 1, numGhosts)
        return expectedValue


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # from searchAgents import mazeDistance
    # food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    pac_pos = currentGameState.getPacmanPosition()
    foodcount = currentGameState.getNumFood()
    currentGameScore = currentGameState.getScore()
    num_capsules = len(currentGameState.getCapsules())

    food = currentGameState.getFood()
    foodsPos = food.asList()
    foodsPos = sorted(foodsPos, key = lambda pos: manhattanDistance(pac_pos, pos))
    closestFoodDist = 0
    if len(foodsPos) > 0:
      closestFoodDist = manhattanDistance(foodsPos[0], pac_pos)

    # evaluate the current state of the ghosts
    nearestGhostDistance = float("inf")
    ghostEval = 0
    for ghost in ghostStates:
      ghostPosition = ghost.getPosition()
      md = manhattanDistance(pac_pos, ghostPosition)
      if ghost.scaredTimer == 0:
        if md < nearestGhostDistance:
          nearestGhostDistance = md
      elif ghost.scaredTimer > md:
        ghostEval += 200 - md

    if nearestGhostDistance == float("inf"):
      nearestGhostDistance = 0
    ghostEval += nearestGhostDistance

    return currentGameScore - 10 * foodcount + 1 * ghostEval + 2 * num_capsules - 2 * closestFoodDist

# Abbreviation
better = betterEvaluationFunction






