import random
import math
import pygame
# The program will ask for an (x, y) coordinate, then return nodes in the tree with that coordinate (multiple nodes bc different paths to same node)
# It will return nodes with highest value, then say which is the best action from there based on simulations. 
# The user should look at nodes with the highest visit count to determine where to move next

# Actions: 1 = Right, 2 = Left, 3 = Up, 4 = Down 

#(0, 0) AT BOTTOM LEFT. (4, 4) AT TOP RIGHT
# Pygame visualization, invalid moves

# Text wrapping
# Show player going through the maze
# Arrows if time

maze = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
path = []
pathPos = []

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
simulationsToRun = 2000
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





# GUI

pygame.init()
screen = pygame.display.set_mode((800, 900))  # taller for buttons
pygame.display.set_caption("Tyler Hogan - MCTS Maze Solver")


player_img = pygame.image.load("Smile2.png")
player_img = pygame.transform.scale(player_img, (100, 100))
# Grid settings
square_size = 100
rows, cols = 5, 5
start_y = 50 #?

# Center the grid horizontally
grid_width = cols * square_size
start_x = (screen.get_width() - grid_width) // 2

# Colors
FILL_COLOR = (0, 0, 0, 100)      # default semi-transparent black
HOVER_COLOR = (0, 100, 255, 120) # hover blue
CLICK_COLOR = (255, 255, 255, 255)   # clicked green
LINE_COLOR = (255, 0, 0)         # red border
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (120, 120, 120)
TEXT_COLOR = (255, 255, 255)

# Track clicked cells
clicked = [[False for _ in range(cols)] for _ in range(rows)]

# Fonts
font = pygame.font.SysFont("dejavuserif", 36)

# Define buttons
button_width, button_height = 320, 50
button_y = 750
button1_rect = pygame.Rect(70, button_y, button_width, button_height)
button2_rect = pygame.Rect(420, button_y, button_width, button_height)

run = True
while run:
    screen.fill((30, 30, 30))
    mouse_pos = pygame.mouse.get_pos()

    # Draw grid
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * square_size
            y = start_y + row * square_size

            # Invisible, but allows for mouse clicking
            rect = pygame.Rect(x, y, square_size, square_size)

            # This allows for highlighting!
            square = pygame.Surface((square_size, square_size), pygame.SRCALPHA) # Makes the square semi-transparent

            # Decide color
            if rect.collidepoint(mouse_pos):
                if clicked[row][col]:
                    square.fill(CLICK_COLOR)
                else:
                    square.fill(HOVER_COLOR)
            else:
                if clicked[row][col]:
                    square.fill(CLICK_COLOR)
                else:
                    square.fill(FILL_COLOR)

            screen.blit(square, (x, y)) #Need blit for surface to place in spot

    # Seperately draw grid lines so that borders don't collide
    grid_height = rows * square_size
    for c in range(cols + 1):
        x = start_x + c * square_size
        pygame.draw.line(screen, LINE_COLOR, (x, start_y), (x, start_y + grid_height), width=3)
    for r in range(rows + 1):
        y = start_y + r * square_size
        pygame.draw.line(screen, LINE_COLOR, (start_x, y), (start_x + grid_width, y), width=3)

    # --- Draw buttons ---
    for rect, text in [(button1_rect, "Generate Maze"),
                       (button2_rect, "Solve Maze")]:
        color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=10) # Filled rectangle
        pygame.draw.rect(screen, LINE_COLOR, rect, width=2, border_radius=10) # The border
        label = font.render(text, True, TEXT_COLOR)
        screen.blit(label, (rect.centerx - label.get_width() // 2,
                            rect.centery - label.get_height() // 2)) # Need blit for text to place in spot

    


    





    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check grid clicks, creates temporary grid of cells and checks over each one to see if they're clicked
            for row in range(rows):
                for col in range(cols):
                    x = start_x + col * square_size
                    y = start_y + row * square_size
                    rect = pygame.Rect(x, y, square_size, square_size)
                    if rect.collidepoint(event.pos):
                        path = []
                        pathPos = []
                        clicked[row][col] = not clicked[row][col]
                        print(f"Clicked cell: ({col}, {4-row})")
                        if(clicked[row][col]):
                            maze[row][col] = 1
                        else:
                            maze[row][col] = 0
                        

            # Check button clicks
            if button1_rect.collidepoint(event.pos):
                print("Randomly generating maze... (placeholder)")
                path = []
                pathPos = []
                maze = generateMaze()    
                while(not is_solvable(maze=maze)):
                    maze = generateMaze()
                for row in range(len(maze)):
                    for col in range(len(maze[row])):
                        if(maze[row][col] == 1):
                            clicked[row][col] = 1
                        else:
                            clicked[row][col] = 0



            elif button2_rect.collidepoint(event.pos):

                if(not is_solvable(maze=maze)):
                    text1 = "Not solveable"

                print("Solving maze...")

                # Render text
                text1 = "Solving Maze..."
                label3 = font.render("Solving Maze...", True, TEXT_COLOR)

                # Position below the maze
                text_x = start_x + (grid_width - label3.get_width()) // 2
                text_y = start_y + row * square_size + 150
                screen.blit(label3, (text_x, text_y))

                screen.blit(player_img, (start_x, 5 * square_size - start_y))

                pygame.display.update()
                if(is_solvable(maze=maze)):
                    root.state = [0, 0]
                    path = []
                    pathPos = []
                    pathPos.append([0, 0])
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
                            path.append("→")
                            
                            root.state[0] += 1
                            if(is_valid_move(root.state)):
                                root.state[0] += 1
                            root.state[0] -= 1
                            pathPos.append(root.state.copy())
                        elif(bestChild.lastAction == 2):
                            print("Move Left: " + str(bestChild.state[0]) + ", " + str(bestChild.state[1]))
                            path.append("←")
                            root.state[0] -= 1
                            if(is_valid_move(root.state)):
                                root.state[0] -= 1
                            root.state[0] += 1
                            pathPos.append(root.state.copy())
                        elif(bestChild.lastAction == 3):
                            print("Move Up: " + str(bestChild.state[0]) + ", " + str(bestChild.state[1]))
                            path.append("↑")
                            root.state[1] += 1
                            if(is_valid_move(root.state)):
                                root.state[1] += 1
                            root.state[1] -= 1
                            pathPos.append(root.state.copy())
                        elif(bestChild.lastAction == 4):
                            print("Move Down: " + str(bestChild.state[0]) + ", " + str(bestChild.state[1]))
                            path.append("↓")
                            root.state[1] -= 1
                            if(is_valid_move(root.state)):
                                root.state[1] -= 1
                            root.state[1] += 1
                            pathPos.append(root.state.copy())
                        print("Root moved to: ", root.state)
                        
                        root.children = []
                        
                        screen.blit(player_img, (start_x + root.state[0] * square_size, start_y + (4-root.state[1]) * square_size))
                        pygame.display.update()
                    


    # Write the path
    pathString = ""
    for direction in path:
        pathString += direction + ", "
    pathString = pathString[:-2]  # remove last comma+space

    # Render text
    text1 = pathString
    if(not is_solvable(maze=maze)):
        text1 = "Not solveable"
    
    label2 = font.render(text1, True, TEXT_COLOR)

    
    # Position below the maze
    text_x = start_x + (grid_width - label2.get_width()) // 2
    text_y = start_y + row * square_size + 150  
    screen.blit(label2, (text_x, text_y))
    
    

    for pos in pathPos:
        x_pixel = start_x + pos[0] * square_size
        y_pixel = start_y + (4 - pos[1]) * square_size  # adjust for bottom-left origin
        screen.blit(player_img, (x_pixel, y_pixel))
        
        
    pygame.display.update()
pygame.quit()

