# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()        
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE *** "
        foodCount = currentGameState.getNumFood()
        ghostPosition = currentGameState.getGhostPosition(1)
        ghostTimer = newScaredTimes.index(0)
        pacmanPosition = newPos        
        pacmanGhostDist = manhattanDistance(pacmanPosition, ghostPosition)
        foodList = newFood.asList()
        
        if pacmanPosition == ghostPosition:            
            if ghostTimer is 0:
                return -100.0
        
        closestFoodDistance = 0
        if len(foodList) > 0:
            #print 'food left: ', len(foodList)
            closest = 100000
            current = 0
            for food in foodList:
                current = manhattanDistance(food,pacmanPosition)
                #print closest
                if current < closest:
                    closest = current                    
            closestFoodDistance = closest

        return -(closestFoodDistance)
        
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
                    
        "*** YOUR CODE HERE ***"        
        initialDepth = 0
        initialAgent = 1        
        bestAction = "stop"
        v = -float("inf")
        legalActions = gameState.getLegalActions(0)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(0, action), initialAgent, initialDepth)             
            if temp > v:
                v = temp
                bestAction = action
        return bestAction       
            
    def value(self, gameState, currentAgent, currentDepth):       
        temp = gameState.getNumAgents()
              
        if currentAgent >= gameState.getNumAgents():
            currentAgent = 0 # roll back to pacman
            currentDepth += 1           
        if currentDepth == self.depth: 
            return self.evaluationFunction(gameState)            
        elif currentAgent is 0: #pacman
            return self.maxValue(gameState, currentAgent, currentDepth)
        else: 
            return self.minValue(gameState, currentAgent, currentDepth)
    
    def maxValue(self, gameState, agent, currentDepth):
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        v = -float("inf")
        legalActions = gameState.getLegalActions(agent)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(agent, action), agent + 1, currentDepth) 
            if temp > v:
                v = temp
        return v
                    
    def minValue(self, gameState, agent, currentDepth):
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        v = float("inf")
        legalActions = gameState.getLegalActions(agent)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(agent, action), agent + 1, currentDepth) 
            if temp < v:
                v = temp
        return v
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
         
        initialDepth = 0
        initialAgent = 1
        
        alpha = -float("inf")
        beta = float("inf")
               
        bestAction = "stop"
        v = -float("inf")
        legalActions = gameState.getLegalActions(0)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(0, action), initialAgent, initialDepth, alpha, beta)             
            if temp > v:
                v = temp
                bestAction = action
            alpha = max(alpha, temp)
        return bestAction       
            
    def value(self, gameState, currentAgent, currentDepth, alpha, beta):
        if currentAgent >= gameState.getNumAgents():
            currentAgent = 0 # roll back to pacman
            currentDepth += 1           
        if currentDepth == self.depth: 
            return self.evaluationFunction(gameState)            
        elif currentAgent is 0: #pacman
            return self.maxValue(gameState, currentAgent, currentDepth, alpha, beta)
        else: 
            return self.minValue(gameState, currentAgent, currentDepth, alpha, beta)
    
    def maxValue(self, gameState, agent, currentDepth, alpha, beta):
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        v = -float("inf")
        legalActions = gameState.getLegalActions(agent)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(agent, action), agent + 1, currentDepth, alpha, beta) 
            if temp > beta:
                return temp
            alpha = max(alpha, temp)
             
            if temp > v:
                v = temp
        return v
                    
    def minValue(self, gameState, agent, currentDepth, alpha, beta):
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        v = float("inf")
        legalActions = gameState.getLegalActions(agent)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(agent, action), agent + 1, currentDepth, alpha, beta) 
            if temp < alpha:
                return temp
            beta = min(beta, temp)
            
            if temp < v:
                v = temp
        return v
        
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
        initialDepth = 0
        initialAgent = 1        
        bestAction = "stop"
        v = -float("inf")
        legalActions = gameState.getLegalActions(0)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(0, action), initialAgent, initialDepth)             
            if temp > v:
                v = temp
                bestAction = action
        return bestAction       
            
    def value(self, gameState, currentAgent, currentDepth):             
        if currentAgent >= gameState.getNumAgents():
            currentAgent = 0 # roll back to pacman
            currentDepth += 1           
        if currentDepth == self.depth: 
            return self.evaluationFunction(gameState)            
        elif currentAgent is 0: #pacman
            return self.maxValue(gameState, currentAgent, currentDepth)
        else: 
            return self.minValue(gameState, currentAgent, currentDepth)
    
    def maxValue(self, gameState, agent, currentDepth):
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        v = -float("inf")
        legalActions = gameState.getLegalActions(agent)        
        for action in legalActions:
            temp = self.value(gameState.generateSuccessor(agent, action), agent + 1, currentDepth) 
            if temp > v:
                v = temp
        return v
                    
    def minValue(self, gameState, agent, currentDepth):
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        v = float("inf")
        legalActions = gameState.getLegalActions(agent) 
        counter = 0
        temp = 0       
        for action in legalActions:
            temp += self.value(gameState.generateSuccessor(agent, action), agent + 1, currentDepth) 
            counter += 1
        return ((1.0*temp)/counter)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <total is regular score to eliminate stop situations, then making sure I don't share the same spot as an angry non scared ghost, and then I am deducting the closest food distance>
    """
    "*** YOUR CODE HERE ***"
    """
    ghostsPositions = [] 
    index = 1    
    for ghost in ghostStates:        
        ghostsPositions.append(ghost.getPosition() )
        index += 1
    print ghostsPositions
    """
    total = currentGameState.getScore()
    #print total
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostTimer = newScaredTimes[0]
    ghostPosition = currentGameState.getGhostPosition(1)
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostPacmanDistance = manhattanDistance(ghostPosition, pacmanPosition)
    foodList = currentGameState.getFood().asList()    
    foodCount = currentGameState.getNumFood()    
        
    if pacmanPosition == ghostPosition:            
        if ghostTimer is 0:
            total = total - 100.0
    
    closestFoodDistance = 0
    if len(foodList) > 0:        
        closest = 100000
        current = 0
        for food in foodList:
            current = manhattanDistance(food,pacmanPosition)            
            if current < closest:
                closest = current                    
        closestFoodDistance = closest
        
    total = total - closestFoodDistance    
        
        
    return total
        
    

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
"""
def mazeDistance(point1, point2, gameState):
    
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
"""