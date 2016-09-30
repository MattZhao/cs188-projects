# hunters.py
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


"""
Busters.py is a vengeful variant of Pacman where Pacman hunts ghosts, but
cannot see them.  Numbers at the bottom of the display are noisy distance
readings to each remaining ghost.

To play your first game, type 'python pacman.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys).  Have fun!
"""
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from game import Configuration
from game import Grid
from util import nearestPoint
from util import manhattanDistance
import sys, util, types, time, random, layout, os

WALL_STRING = '%'
RED_WALL_STRING = 'R'
BLUE_WALL_STRING = 'B'
EMPTY_SQUARE_STRING = ' '

GHOST_COLLISION_REWARD = -500
WON_GAME_REWARD = 500

###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    Note that in classic Pacman, Pacman is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def getLegalActions( self, agentIndex=0 ):
        """
        Returns the legal actions for the agent specified.
        """
        if self.isWin() or self.isLose(): return []

        if agentIndex == 0:  # Pacman is moving
            return PacmanRules.getLegalActions( self )
        else:
            return GhostRules.getLegalActions( self, agentIndex )

    def generateSuccessor( self, agentIndex, action):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isWin() or self.isLose(): raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = GameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agentIndex == 0:  # Pacman is moving
            state.data._eaten = [False for i in range(state.getNumAgents())]
            PacmanRules.applyAction( state, action )
        else:                # A ghost is moving
            GhostRules.applyAction( state, action, agentIndex )

        # Time passes
        if agentIndex == 0:
            state.data.scoreChange += -TIME_PENALTY # Penalty for waiting around

        # Resolve multi-agent effects
        GhostRules.checkDeath( state, agentIndex )

        # Book keeping
        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange

        # Pacman observes neighboring squares
        state.getObservation()

        return state

    def getLegalPacmanActions( self ):
        return self.getLegalActions( 0 )

    def generatePacmanSuccessor( self, action ):
        """
        Generates the successor state after the specified pacman move
        """
        return self.generateSuccessor( 0, action )

    def getPacmanState( self ):
        """
        Returns an AgentState object for pacman (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agentStates[0].copy()

    def getPacmanPosition( self ):
        return self.data.agentStates[0].getPosition()

    def getNumAgents( self ):
        return len( self.data.agentStates )

    def getScore( self ):
        return self.data.score

    def isLose( self ):
        return self.data._lose

    def isWin( self ):
        return self.data._win

    ################################
    # Additions for Hunters Pacman #
    ################################

    def getObservation(self):
        x, y = self.getPacmanPosition()
        adjacent = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        for x, y in adjacent:
            self.data.observedPositions[x][y] = True
        return { (x, y) : self.squareInfo(x, y) for x, y in adjacent }

    def squareInfo(self, x, y):
        if self.hasWall(x, y):
            return WALL_STRING
        elif self.data.layout.redWalls[x][y]:
            return RED_WALL_STRING
        elif self.data.layout.blueWalls[x][y]:
            return BLUE_WALL_STRING
        else:
            return EMPTY_SQUARE_STRING

    def getObservedPositions(self):
        return self.data.observedPositions

    def getHouseWalls(self, house):
        return layout.buildHouseAroundCenter(*house)

    def getPossibleHouses(self):
        return layout.pickPossibleLocations(self.data.layout.width, self.data.layout.height)

    def getEvidence(self):
        evidence = {}
        possible = {w for h in self.getPossibleHouses() for w in self.getHouseWalls(h)}
        for pos in self.getObservedPositions().asList():
            if pos in possible:
                evidence[pos] = self.squareInfo(*pos)
        return evidence

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__( self, prevState = None ):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState != None:
            self.data = GameStateData(prevState.data)
            self.numMoves = prevState.numMoves
            self.maxMoves = prevState.maxMoves
        else: # Initial state
            self.data = GameStateData()
            self.numMoves = 0
            self.maxMoves = -1

    def deepCopy( self ):
        state = GameState( self )
        state.data = self.data.deepCopy()
        return state

    def __eq__( self, other ):
        """
        Allows two states to be compared.
        """
        return self.data == other.data

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.
        """
        return hash( str( self ) )

    def __str__( self ):

        return str(self.data)

    def initialize( self, layout, numGhostAgents=1000 ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(layout, numGhostAgents)

    def getGhostPosition( self, agentIndex ):
        if agentIndex == 0:
            raise "Pacman's index passed to getGhostPosition"
        return self.data.agentStates[agentIndex].getPosition()

    def getGhostState( self, agentIndex ):
        if agentIndex == 0:
            raise "Pacman's index passed to getGhostPosition"
        return self.data.agentStates[agentIndex]

    def getCapsules(self):
        """
        Returns a list of positions (x,y) of the remaining capsules.
        """
        return self.data.capsules

    def getNumFood( self ):
        return self.data.food.count()

    def getFood(self):
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """
        return self.data.food

    def getWalls(self):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        walls = state.getWalls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def hasFood(self, x, y):
        return self.data.food[x][y]

    def hasWall(self, x, y):
        return self.data.layout.walls[x][y]

############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

COLLISION_TOLERANCE = 0.7 # How close ghosts must be to Pacman to kill
TIME_PENALTY = 1 # Number of points lost each round

class HuntersGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def newGame( self, layout, pacmanAgent, ghostAgents, display, maxMoves= -1 ):
        agents = [pacmanAgent] + ghostAgents
        initState = GameState()
        initState.initialize( layout, len(ghostAgents) )
        game = Game(agents, display, self)
        game.state = initState
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.isWin(): self.win(state, game)
        if state.isLose(): self.lose(state, game)

    def win( self, state, game ):
        print "Pacman found the food! Score: %d" % state.data.score
        game.gameOver = True

    def lose( self, state, game ):
        print "Uh oh, You found the Princess! Wraaaang she's in another house go away. Score: %d" % state.data.score
        game.gameOver = True

class PacmanRules:
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """
    def getLegalActions( state ):
        """
        Returns a list of possible actions.
        """
        return Actions.getPossibleActions( state.getPacmanState().configuration, state.data.layout.walls, state.data.layout.redWalls, state.data.layout.blueWalls )
    getLegalActions = staticmethod( getLegalActions )

    def applyAction( state, action ):
        """
        Edits the state to reflect the results of the action.
        """
        legal = PacmanRules.getLegalActions( state )
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        pacmanState = state.data.agentStates[0]

        # Update Configuration
        vector = Actions.directionToVector( action, 1 )
        pacmanState.configuration = pacmanState.configuration.generateSuccessor( vector )

        # Eat
        next = pacmanState.configuration.getPosition()
        nearest = nearestPoint( next )
        if manhattanDistance( nearest, next ) <= 0.5 :
            # Remove food
            PacmanRules.consume( nearest, state )

    applyAction = staticmethod( applyAction )

    def consume( position, state ):
        x,y = position
        # Eat food
        if state.data.food[x][y]:
            state.data.scoreChange += 10
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position

            numFood = state.getNumFood()
            if numFood == 0 and not state.data._lose:
                state.data.scoreChange += WON_GAME_REWARD
                state.data._win = True

    consume = staticmethod( consume )

class GhostRules:
    """
    These functions dictate how ghosts interact with their environment.
    """
    def applyAction( state, action, ghostIndex):
        if action != Directions.STOP:
            raise Exception("Illegal ghost action " + str(action))

        ghostState = state.data.agentStates[ghostIndex]
        vector = Actions.directionToVector( action, 1 )
        ghostState.configuration = ghostState.configuration.generateSuccessor( vector )
    applyAction = staticmethod( applyAction )

    def checkDeath( state, agentIndex):
        pacmanPosition = state.getPacmanPosition()
        if agentIndex == 0: # Pacman just moved; Anyone can kill him
            for index in range( 1, len( state.data.agentStates ) ):
                ghostState = state.data.agentStates[index]
                ghostPosition = ghostState.configuration.getPosition()
                if GhostRules.canKill( pacmanPosition, ghostPosition ):
                    GhostRules.collide( state, ghostState, index )
        else:
            ghostState = state.data.agentStates[agentIndex]
            ghostPosition = ghostState.configuration.getPosition()
            if GhostRules.canKill( pacmanPosition, ghostPosition ):
                GhostRules.collide( state, ghostState, agentIndex )
    checkDeath = staticmethod( checkDeath )

    def collide( state, ghostState, agentIndex):
        state.data.scoreChange += GHOST_COLLISION_REWARD
        state.data._lose = True

    collide = staticmethod( collide )

    def canKill( pacmanPosition, ghostPosition ):
        return manhattanDistance( ghostPosition, pacmanPosition ) <= COLLISION_TOLERANCE
    canKill = staticmethod( canKill )

    def placeGhost(state, ghostState):
        ghostState.configuration = ghostState.start
    placeGhost = staticmethod( placeGhost )

#############################
# FRAMEWORK TO START A GAME #
#############################

def default(str):
    return str + ' [Default: %default]'

def parseAgentArgs(str):
    if str == None: return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key,val = p, 1
        opts[key] = val
    return opts

def readCommand( argv ):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python busters.py <options>
    EXAMPLE:    python busters.py --layout bigHunt
                  - starts an interactive game on a big board
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='treasureHunt')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-a','--agentArgs',dest='agentArgs',
                      help='Comma seperated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghostAgents module to use'),
                      metavar = 'TYPE', default='StationaryGhostAgent')
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-k', '--numghosts', type='int', dest='numGhosts',
                      help=default('The maximum number of ghosts to use'), default=1)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-d', '--displayGhosts', action='store_true', dest='displayGhosts',
                      help='Renders the ghosts in the display (cheating)', default=False)
    parser.add_option('-t', '--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_option('-r', '--randomBoard', action='store_true', dest='randomBoard',
                      help='Generates some random board', default=False)
    parser.add_option('-v', '--vpiBoard', action='store_true', dest='vpiBoard',
                      help='Generates a special board for the VPI sub-problem',
                      default=False)
    parser.add_option('-s', '--seed', type='str', dest='seed',
                      help=default('Generates a random board using the specified seed'), default=None)

    options, otherjunk = parser.parse_args()
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + otherjunk)
    args = dict()

    if options.randomBoard or options.seed:
        # options.seed defaults to None which uses system time
        args['layout'] = layout.Layout(options.seed)
    elif options.vpiBoard:
        args['layout'] = layout.Layout(options.seed, vpi=True)
    else:
        # Choose a layout
        args['layout'] = layout.getLayout( options.layout )
        if args['layout'] == None: raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a ghost agent
    ghostType = loadAgent(options.ghost, options.quietGraphics)
    args['ghosts'] = [ghostType( i+1 ) for i in range( options.numGhosts )]

    # Choose a Pacman agent
    noKeyboard = options.quietGraphics
    pacmanType = loadAgent(options.pacman, noKeyboard)
    agentOpts = parseAgentArgs(options.agentArgs)
    # agentOpts['ghostAgents'] = args['ghosts']
    pacman = pacmanType(**agentOpts) # Instantiate Pacman with agentArgs
    args['pacman'] = pacman

    import graphicsDisplay
    args['display'] = graphicsDisplay.FirstPersonPacmanGraphics(options.zoom, options.displayGhosts, frameTime = options.frameTime, hunters=True)
    args['numGames'] = options.numGames

    return args

def loadAgent(pacman, nographics):
    # Looks through all pythonPath Directories for the right module,
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        pythonPathDirs = pythonPathStr.split(':')
    else:
        pythonPathDirs = pythonPathStr.split(';')
    pythonPathDirs.append('.')

    for moduleDir in pythonPathDirs:
        if not os.path.isdir(moduleDir): continue
        moduleNames = [f for f in os.listdir(moduleDir) if f.endswith('gents.py')]
        for modulename in moduleNames:
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if pacman in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception('Using the keyboard requires graphics (not text display)')
                return getattr(module, pacman)
    raise Exception('The agent ' + pacman + ' is not specified in any *Agents.py.')

def runGames( layout, pacman, ghosts, display, numGames, maxMoves=-1):
    # Hack for agents writing to the display
    import __main__
    __main__.__dict__['_display'] = display

    rules = HuntersGameRules()
    games = []

    for i in range( numGames ):
        game = rules.newGame( layout, pacman, ghosts, display, maxMoves )
        game.run()
        games.append(game)

    if numGames > 1:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True)/ float(len(wins))
        print 'Average Score:', sum(scores) / float(len(scores))
        print 'Scores:       ', ', '.join([str(score) for score in scores])
        print 'Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), winRate)
        print 'Record:       ', ', '.join([ ['Loss', 'Win'][int(w)] for w in wins])

    return games

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    args = readCommand( sys.argv[1:] ) # Get game components based on input
    runGames( **args )
