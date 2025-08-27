import jax
import jax.numpy as jnp
import mctx
import random
from mctx import RootFnOutput, RecurrentFnOutput
import pygame


# The program takes the startPos variable with the coordinates in the first two indices and returns the next best move and probability distribution

#ACTION: 0->DOWN, 1->UP, 2->RIGHT, 3->LEFT

#(0, 0) AT TOP LEFT. (4, 4) AT BOTTOM 
#COORDINATES ARE (y, x), since its row x col

useRandomMaze = False
maze = jnp.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])
numSimulations = 100
# MAZE GENERATOR CODE 

def generateMaze():
    availableTiles = [0, 1]
    maze = jnp.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    count = 0 #So maze 0, 0 and 4, 4 stay 0
    while(count < 25):
        action = random.choice(availableTiles)
        maze = maze.at[count // 5, count % 5].set(action)
        count += 1
    return maze


# Recursive DFS function
def is_solvable(maze):
    ro = 5
    co = 5
    start = [0, 0]
    goal = [ro-1, co-1]

    if maze[start[0], start[1]] == 1 or maze[goal[0], goal[1]] == 1:
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
            if 0 <= nr < ro and 0 <= nc < co:  # in bounds
                if maze[nr, nc] == 0 and (nr, nc) not in visited:
                    if dfs(nr, nc):
                        return True
        return False

    return dfs(start[0], start[1])


# MCTS CODE

goal = [4, 4]
path = []
pathPos = []


def is_valid_tile(embedding):
    row = embedding[0, 0].astype(int)
    col = embedding[0, 1].astype(int)
    in_bounds = (row >= 0) & (row < 5) & (col >= 0) & (col < 5)
    not_wall = jnp.where(in_bounds, maze[row, col] == 0, False)
    return not_wall & in_bounds
    

startPos = jnp.array([[0, 0, 0]])
downEmbeding = startPos.at[0, 0].add(1)
upEmbeding = startPos.at[0, 0].add(-1)
rightEmbeding = startPos.at[0, 1].add(1)
leftEmbeding = startPos.at[0, 1].add(-1)
rng_key = jax.random.PRNGKey(0)

def noisy_prior(prior_logits, rng_key, alpha=1, frac=0.00): 
    noise = jax.random.dirichlet(rng_key, jnp.full(prior_logits.shape[-1], alpha))
    p = jax.nn.softmax(prior_logits)
    mixed = (1 - frac) * p + frac * noise
    return jnp.log(mixed)





# Next state predictor function
def recurrent_fn(params, rng_key, action, embedding):
    # jax.debug.print("StartPos = {}", startPos)
    # jax.debug.print("Embedding = {}", embedding)

    new_embedding = embedding
    new_embedding = jnp.where(action == 0, embedding.at[0, 0].add(1), new_embedding) # Down
    new_embedding = jnp.where(action == 1, embedding.at[0, 0].add(-1), new_embedding) # Up
    new_embedding = jnp.where(action == 2, embedding.at[0, 1].add(1), new_embedding) # Right
    new_embedding = jnp.where(action == 3, embedding.at[0, 1].add(-1), new_embedding) # Left
    is_terminal = (new_embedding[0,0].astype(int) == goal[0]) & (new_embedding[0,1].astype(int) == goal[1])
    valid = is_valid_tile(new_embedding)
    # jax.debug.print("New embedding, Valid = {}, {}", embedding, valid)

    new_embedding = jnp.where(valid, new_embedding, embedding)
    new_embedding = jnp.where(is_terminal, new_embedding.at[0, 2].set(1), new_embedding) # Left
    
    #jax.debug.print("embedding = {}", new_embedding)

    terminated = new_embedding[0, 2].astype(int)

    reward = jnp.where(terminated == 1, 1000, -((goal[0] - new_embedding[0, 0].astype(int)) + (goal[1]- new_embedding[0, 1].astype(int)))/5.0)
    reward = jnp.where((~valid) & (terminated == 0), -100.0, reward)
    reward = reward.flatten()  # Flatten to ensure shape (1,) not (1,1), 1D not 1x1 in 2D
    discount = jnp.where(terminated == 1, jnp.array([0]), jnp.array([1])) # Rest are 0 if we finish


    
    downEmbeding = new_embedding.at[0, 0].add(1)
    upEmbeding = new_embedding.at[0, 0].add(-1)
    rightEmbeding = new_embedding.at[0, 1].add(1)
    leftEmbeding = new_embedding.at[0, 1].add(-1)

    valid_mask = jnp.array([ # Checks if is a valid tile and not terminated. Updates logits with this
        is_valid_tile(downEmbeding) & (terminated != 1),   # Down
        is_valid_tile(upEmbeding) & (terminated != 1),  # Up
        is_valid_tile(rightEmbeding) & (terminated != 1),   # Right
        is_valid_tile(leftEmbeding) & (terminated != 1),  # Left jnp.array([[0,-1, 0]])
    ])

    prior_logits = jnp.where(
        valid_mask[None,:],
        jnp.zeros((1,4)),     # 0 logits for valid moves
        -jnp.inf              # mask invalid moves
    )

    value = -((goal[0] - new_embedding[0, 0].astype(int)) + (goal[1]- new_embedding[0, 1].astype(int)))
    value = value.flatten() # Flatten to ensure shape (1,) not (1,1), 1D not 1x1 in 2D
    


    
    output = RecurrentFnOutput(
        reward=reward,      
        discount=discount,  
        prior_logits=prior_logits,  
        value=value        
    )
    return output, new_embedding

params = None  




    




#path.append(policy_output.action.item())

def moveNextAction(policy_output, startPos, path):
    jax.debug.print("Next best move is: {}", policy_output.action.item())
    greedy  = int(jnp.argmax(policy_output.action_weights))
    jax.debug.print("Next best greedy move is: {}", greedy)
    print("Action probabilities:", policy_output.action_weights)
    
    action = greedy #policy_output.action.item()

    if action == 0:
        path.append("↓")
        startPos = startPos.at[0, 0].add(1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 0].add(1)
        startPos = startPos.at[0, 0].add(-1)
        pathPos.append([startPos[0, 0], startPos[0, 1]])
    elif action == 1:
        path.append("↑")
        startPos = startPos.at[0, 0].add(-1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 0].add(-1)
        startPos = startPos.at[0, 0].add(1)
        pathPos.append([startPos[0, 0], startPos[0, 1]])
    elif action == 2:
        path.append("→")
        startPos = startPos.at[0, 1].add(1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 1].add(1)
        startPos = startPos.at[0, 1].add(-1)
        pathPos.append([startPos[0, 0], startPos[0, 1]])
    elif action == 3:
        path.append("←")
        startPos = startPos.at[0, 1].add(-1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 1].add(-1)
        startPos = startPos.at[0, 1].add(1)
        pathPos.append([startPos[0, 0], startPos[0, 1]])

    jax.debug.print("Position: {}", startPos)
    return startPos




#print("Action probabilities:", policy_output.action_weights)


# GUI


pygame.init()
screen = pygame.display.set_mode((800, 900))  # taller for buttons
pygame.display.set_caption("Tyler Hogan - MCTS Maze Solver")
# Fonts
font = pygame.font.SysFont("dejavuserif", 36)


class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('gray')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.txt_surface = font.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active if clicked
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print("Entered text:", self.text)  # Do something with the text
                    global numSimulations
                    numSimulations = int(self.text)
                    print(numSimulations)
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render text
                self.txt_surface = font.render(self.text, True, (255, 255, 255))

    def update(self):
        # Resize box if text is too long
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Draw text
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Draw rect
        pygame.draw.rect(screen, self.color, self.rect, 2)







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



# Define buttons
button_width, button_height = 320, 50
button_y = 750
button1_rect = pygame.Rect(70, button_y, button_width, button_height)
button2_rect = pygame.Rect(420, button_y, button_width, button_height)

input_box = InputBox(300, 850, 200, 40)
clock = pygame.time.Clock()


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

     # --- Input Box ---

    


    





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
                        print(f"Clicked cell: ({row}, {col})")
                        if(clicked[row][col]):
                            maze = maze.at[row, col].set(1)
                        else:
                            maze = maze.at[row, col].set(0)
                        print("Button clicked, new maze:")
                        for r in range(5):
                            row_str = " ".join(str(maze[r, c]) for c in range(5))
                            print(row_str)
                        print()
                        

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

                screen.blit(player_img, (start_x, start_y))

                pygame.display.update()

                # Solve the maze
                if(is_solvable(maze=maze)):
                    # Run MCTS search for 0, 0
                    startPos = jnp.array([[0, 0, 0]])
                    path = []
                    pathPos = []
                    downEmbeding = startPos.at[0, 0].add(1)
                    upEmbeding = startPos.at[0, 0].add(-1)
                    rightEmbeding = startPos.at[0, 1].add(1)
                    leftEmbeding = startPos.at[0, 1].add(-1)
                    print(is_valid_tile(upEmbeding))
                    print(is_valid_tile(leftEmbeding))
                    print(is_valid_tile(downEmbeding))
                    print(is_valid_tile(rightEmbeding))
                    root = RootFnOutput(
                        
                        # Vector with important information, not exactly the raw state
                        
                        prior_logits=jnp.array([[jnp.where(is_valid_tile(downEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(upEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(rightEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(leftEmbeding), 0.0, -jnp.inf)]]),  # Estimated Policy
                        value=jnp.array([0.0]),                      # shape (1,)
                        embedding=startPos #jnp.zeros((1, 3))                  
                    )

                    invalid_actions = jnp.array([
                        not is_valid_tile(downEmbeding),
                        not is_valid_tile(upEmbeding),
                        not is_valid_tile(rightEmbeding),
                        not is_valid_tile(leftEmbeding),
                    ], dtype=bool)

                    # Add batch dimension
                    invalid_actions = invalid_actions[None, :]  # shape becomes (1, 4)

                    policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations=numSimulations, invalid_actions=invalid_actions)
                    pathPos.append(jnp.array([0, 0]))
                    
                    
                    while(not (startPos[0, 0].astype(int) == goal[0] and startPos[0, 1].astype(int) == goal[1])):
                        startPos = moveNextAction(policy_output=policy_output, startPos = startPos, path = path)
                        downEmbeding = startPos.at[0, 0].add(1)
                        upEmbeding = startPos.at[0, 0].add(-1)
                        rightEmbeding = startPos.at[0, 1].add(1)
                        leftEmbeding = startPos.at[0, 1].add(-1)

                        rng_key, subkey = jax.random.split(rng_key)

                        root = RootFnOutput(
                        
                        prior_logits=noisy_prior(jnp.array([[jnp.where(is_valid_tile(downEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(upEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(rightEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(leftEmbeding), 0.0, -jnp.inf)]]), subkey),  # Estimated Policy
                        value=jnp.array([0.0]),                      
                        embedding=startPos #jnp.zeros((1, 3))                  
                        )

                        screen.blit(player_img, (start_x + startPos[0, 1].astype(int) * square_size, start_y + startPos[0, 0].astype(int) * square_size))
                        pygame.display.update()
                        invalid_actions = []
                        invalid_actions = jnp.array([
                        not is_valid_tile(downEmbeding),
                        not is_valid_tile(upEmbeding),
                        not is_valid_tile(rightEmbeding),
                        not is_valid_tile(leftEmbeding),
                        ], dtype=bool)

                        # Add batch dimension
                        invalid_actions = invalid_actions[None, :]  # shape becomes (1, 4)

                        policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations=numSimulations, invalid_actions=invalid_actions) # Update invalid actions
                        
                    print("Path: ", path) 
        input_box.handle_event(event)
                


    

    input_box.update()
    input_box.draw(screen)                
    label_sim = font.render("Simulations:", True, TEXT_COLOR)
    screen.blit(label_sim, (input_box.rect.x - label_sim.get_width() - 10, input_box.rect.y ))
    

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
        x_pixel = start_x + pos[1] * square_size
        y_pixel = start_y + (pos[0]) * square_size  # adjust for bottom-left origin
        screen.blit(player_img, (x_pixel, y_pixel))
        
    pygame.display.flip()
    clock.tick(30)
    #pygame.display.update()
pygame.quit()

