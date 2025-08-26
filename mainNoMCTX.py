import random
import math
import pygame
# The program will ask for an (x, y) coordinate, then return nodes in the tree with that coordinate (multiple nodes bc different paths to same node)
# It will return nodes with highest value, then say which is the best action from there based on simulations. 
# The user should look at nodes with the highest visit count to determine where to move next

# Actions: 1 = Right, 2 = Left, 3 = Up, 4 = Down 

#(0, 0) AT BOTTOM LEFT. (4, 4) AT TOP RIGHT
# Pygame visualization, invalid moves

pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

run = True
gridSquare = pygame.Rect((100, 100, 200, 200))
while run:
    screen.fill((30, 30, 30))  

    # Border only: width > 0
    pygame.draw.rect(screen, (255, 0, 0), gridSquare, width=5)

    # Filled rectangle (semi-transparent)
    square = pygame.Surface((200, 200), pygame.SRCALPHA)
    square.fill((0, 0, 0, 100))  # Green with transparency
    screen.blit(square, (100, 100))


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.update()
pygame.quit()


























useRandomMaze = True
maze = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

# MAZE GENERATOR CODE
def generateMaze():
    availableTiles = [0, 1]
    maze = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    count = 0 #So maze 0, 0 and 4, 4 stay 0
    while(count < 25):
        action = random.choice(availableTiles)
        maze[(count // 5)][(count) % 5] = action
        count += 1
    return maze


# Recursive DFS function
def is_solvable(maze):
    rows = len(maze)
    cols = len(maze[0])
    start = [4, 0]
    goal = [0, cols-1]

    if maze[start[0]][start[1]] == 1 or maze[goal[0]][goal[1]] == 1:
        return False  # blocked start or goal

    visited = set()

    def dfs(r, c):
        
        if r == goal[0] and c == goal[1]:
            return True
        visited.add((r, c))
        #print(visited)
        # Possible moves: up, down, left, right
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        for dr, dc in directions:
            nr = r + dr
            nc =  c + dc
            if 0 <= nr < rows and 0 <= nc < cols:  # in bounds
                if maze[nr][nc] == 0 and (nr, nc) not in visited:
                    if dfs(nr, nc):
                        return True
        return False

    return dfs(start[0], start[1])

    

# MCTS CODE

def is_valid_move(state):
    x = state[0]
    y = state[1]
    if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[len(maze[0])-y-1][x] == 0:
        return True
    return False


class Node:
    def __init__(self, state, lastAction, parent=None):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.reward = 0
        self.lastAction = lastAction

        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0
    
    # Potentially terminating factor

root = Node(state=[0, 0], lastAction=0)
simulationsToRun = 10000
horizonLimit = 100
simulations = 0



def selection(node):
    #print("Running selection on " + str(node.state[0]) + " " + str(node.state[1]))
    node.visits += 1

    if(len(node.children) < 4):
        #Choose one to expand into
        availableActions = [1, 2, 3, 4] # 7 right, 8 up
        
        for child in node.children:
            availableActions.remove(child.lastAction)
            #print("CHILD STATE: " + str(child.state[0]) + ", " + str(child.state[1]) + " | Last Action: " + str(child.lastAction))
        action = random.choice(availableActions)  # 1 = Right, 2 = Left, 3 = Up, 4 = Down 

        
        expansion(node, action)
    else:
        #Follow Formula
        nextExploration = -100
        nextExplorationChild = None
        for child in node.children:
                UCB1 = (child.reward/child.visits) + (3000) * (math.sqrt(math.log(node.visits)/child.visits)) #Periodic exploration? abs(math.sin(simulations))
                if(UCB1 > nextExploration):
                    nextExploration = UCB1
                    nextExplorationChild = child
                
        selection(nextExplorationChild)
    
    

def expansion(node, action):
    
    #print("Expansion on " + str(node.state[0]) + " " + str(node.state[1]))
    newState = node.state.copy()
    newState[0] += 1
    if (action == 1 ) and (is_valid_move(newState)): 
        newState[0] += 1
    newState[0] -= 1
    newState[0] -= 1
    if (action == 2) and (is_valid_move(newState)): 
        newState[0] -= 1
    newState[0] += 1
    newState[1] += 1
    if (action == 3) and (is_valid_move(newState)): 
        newState[1] += 1
    newState[1] -= 1
    newState[1] -= 1
    if (action == 4) and (is_valid_move(newState)): 
        newState[1] -= 1
    newState[1] += 1
    newChild = Node(state=newState, lastAction=action, parent=node)
    node.add_child(newChild)
    simulation(newChild)


def simulation(node):
    #print("Simulation")

    newState = node.state.copy()
    rewardOfSimulation = 0
    discount = 1
    horizon = 0
    horizonLimitBroken = False
    #print("Starting state: " + str(newState[0]) + ", " + str(newState[1]))
    stateSearchPath = []
    while newState != [4, 4]:
        randomAction = random.randint(1, 4)
        newState[0] += 1
        if ((randomAction == 1 and is_valid_move(newState))or randomAction == 6 or randomAction == 7 or randomAction == 9 or randomAction == 10 or randomAction == 11 or randomAction == 12 or randomAction == 13 or randomAction == 14) and (newState[0] < 5): 
            newState[0] += 1
        newState[0] -= 1
        newState[0] -= 1
        if (randomAction == 2 and is_valid_move(newState)) and (newState[0] > 0):
            newState[0] -= 1
        newState[0] += 1
        newState[1] += 1
        if ((randomAction == 3 and is_valid_move(newState)) or randomAction == 5 or randomAction == 8 or randomAction == 15 or randomAction == 16 or randomAction == 17 or randomAction == 18 or randomAction == 19 or randomAction == 20) and (newState[1] < 5):
            newState[1] += 1
        newState[1] -= 1
        newState[1] -= 1
        if (randomAction == 4 and is_valid_move(newState)) and (newState[1] > 0):
            newState[1] -= 1
        newState[1] += 1
        stateSearchPath.append(newState.copy())
        rewardOfSimulation -= ((5-newState[0] + 5-newState[1])/5) * discount
        discount *= 0.99
        horizon += 1
        if horizon > horizonLimit:
            horizonLimitBroken = True
            break

    if not horizonLimitBroken:
        rewardOfSimulation += 1000
    # Print entire search path on one line
    #print("Search path: " + str(stateSearchPath))
    backpropagation(node, rewardOfSimulation)
    

    
def backpropagation(node, rewardOfSimulation):
    #print("Backpropagation")
    global simulations 

    while(node.parent != None):
        node.reward += rewardOfSimulation
        node.visits += 1
        node = node.parent
    node.visits += 1
    node.reward += rewardOfSimulation
    simulations += 1



            

if __name__ == "__main__":
    path = []

    if(useRandomMaze):
        maze = generateMaze()    
        while(not is_solvable(maze=maze)):
            maze = generateMaze()
        for row in maze:
            print(row)


    while(root.state != [4, 4]):
        simulations = 0
        while(simulations < simulationsToRun):
            selection(root)

        
        bestAverageReward = -100000
        bestChild = Node([-5, -5], 0)
        for child in root.children:
            print("Child reward " + str(child.state[0]) + ", " + str(child.state[1]) + ": " + str(child.reward/child.visits) + " | Visits: " + str(child.visits))
            if (child.reward/child.visits) > bestAverageReward:
                bestAverageReward = (child.reward/child.visits)
                bestChild = child


        if(bestChild.lastAction == 1):
            print("Move Right: " + str(bestChild.state[0]) + ", " + str(bestChild.state[1]))
            path.append("Right")
            root.state[0] += 1
            if(is_valid_move(root.state)):
                root.state[0] += 1
            root.state[0] -= 1
        elif(bestChild.lastAction == 2):
            print("Move Left: " + str(bestChild.state[0]) + ", " + str(bestChild.state[1]))
            path.append("Left")
            root.state[0] -= 1
            if(is_valid_move(root.state)):
                root.state[0] -= 1
            root.state[0] += 1
        elif(bestChild.lastAction == 3):
            print("Move Up: " + str(bestChild.state[0]) + ", " + str(bestChild.state[1]))
            path.append("Up")
            root.state[1] += 1
            if(is_valid_move(root.state)):
                root.state[1] += 1
            root.state[1] -= 1
        elif(bestChild.lastAction == 4):
            print("Move Down: " + str(bestChild.state[0]) + ", " + str(bestChild.state[1]))
            path.append("Down")
            root.state[1] -= 1
            if(is_valid_move(root.state)):
                root.state[1] -= 1
            root.state[1] += 1
        print("Root moved to: ", root.state)
        root.children = []
    print("Path: ", path) 




