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
      A reflex Agent chooses an move at each choice point by examining
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
        scores = [self.evaluationFunction(gameState, move) for move in legalMoves]
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
        successorgameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorgameState.getPacmanPosition()
        newFood = successorgameState.getFood()
        newGhostStates = successorgameState.getGhostStates()
        foodNum = currentGameState.getFood().count()
        if len(newFood.asList()) == foodNum:  # if this action does not eat a food 
            dis = float("inf")
            for food in newFood.asList():
                if manhattanDistance(food , newPos) < dis :
                    dis = manhattanDistance(food, newPos)
        else:
            dis = 0
        for ghost in newGhostStates:  # the impact of ghost surges as distance get close
            dis += 4 ** (2 - manhattanDistance(ghost.getPosition(), newPos))
        return -dis

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
      multi-Agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always Agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your Minimax Agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the Minimax move from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing Minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an Agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, move):
            Returns the successor game state after an Agent takes an move

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def Minimax(Agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if Agent == 0:
                return max(Minimax(1, depth, gameState.generateSuccessor(Agent, newState)) for newState in gameState.getLegalActions(Agent))
            else: 
                nextAgent = Agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                   depth += 1
                return min(Minimax(nextAgent, depth, gameState.generateSuccessor(Agent, newState)) for newState in gameState.getLegalActions(Agent))
        
        Maximum = -999999
        move = Directions.WEST
        for pacmacMove in gameState.getLegalActions(0):
            _temp = Minimax(1, 0, gameState.generateSuccessor(0, pacmacMove))
            if _temp > Maximum or Maximum == float("-inf"):
                Maximum = _temp
                move = pacmacMove
        return move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your Minimax Agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the Minimax move using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Maximazer(Agent, depth, currentState, a, b):
            val = float("-inf")
            for newState in currentState.getLegalActions(Agent):
                val = max(val, alphaBetaPruning(1, depth, currentState.generateSuccessor(Agent, newState), a, b))
                if val > b:
                    return val
                a = max(a, val)
            return val

        def Minimizer(Agent, depth, currentState, a, b):
            val = float("inf")

            nextAgent = Agent + 1 
            if currentState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1

            for newState in currentState.getLegalActions(Agent):
                val = min(val, alphaBetaPruning(nextAgent, depth, currentState.generateSuccessor(Agent, newState), a, b))
                if val < a:
                    return val
                b = min(b, val)
            return val

        def alphaBetaPruning(Agent, depth, currentState, a, b):
            if currentState.isLose() or currentState.isWin() or depth == self.depth:
                return self.evaluationFunction(currentState)

            if Agent == 0:
                return Maximazer(Agent, depth, currentState, a, b)
            elif (Agent >= 1):
                return Minimizer(Agent, depth, currentState, a, b)

        _temp = float("-inf")
        move = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")
        pacmanMoves = gameState.getLegalActions(0)
        for pacmacMove in pacmanMoves:
            ghostVal = alphaBetaPruning(1, 0, gameState.generateSuccessor(0, pacmacMove), alpha, beta)
            if ghostVal > _temp:
                _temp = ghostVal
                move = pacmacMove
            if _temp > beta:
                return _temp
            alpha = max(alpha, _temp)

        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectiMax Agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectiMax move using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def expectiMax(Agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth: 
                return self.evaluationFunction(gameState)
            if Agent == 0: 
                return max(expectiMax(1, depth, gameState.generateSuccessor(Agent, newState)) for newState in gameState.getLegalActions(Agent))
            else:  
                nextAgent = Agent + 1 
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return sum(expectiMax(nextAgent, depth, gameState.generateSuccessor(Agent, newState)) for newState in gameState.getLegalActions(Agent)) / float(len(gameState.getLegalActions(Agent)))

        Maximum = float("-inf")
        move = Directions.STOP
        for pacmacMove in gameState.getLegalActions(0):
            _temp = expectiMax(1, 0, gameState.generateSuccessor(0, pacmacMove))
            if _temp > Maximum or Maximum == float("-inf"):
                Maximum = _temp
                move = pacmacMove

        return move

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    nearestFood = -1
    for food in newFoodList:
        distance = util.manhattanDistance(newPos, food)
        if nearestFood >= distance or nearestFood == -1:
            nearestFood = distance

    sumDisToGhost = 1
    numOfSuccessorGhosts = 0
    for ghostCoord in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghostCoord)
        sumDisToGhost += distance
        if distance <= 1:
            numOfSuccessorGhosts += 1

    
    newCapsule = currentGameState.getCapsules()
    numOfCaptules = len(newCapsule)

    return currentGameState.getScore() + (1 / float(nearestFood)) - (1 / float(sumDisToGhost)) - numOfSuccessorGhosts - numOfCaptules

# Abbreviation
better = betterEvaluationFunction

