import jax
import jax.numpy as jnp
import mctx
import random
from mctx import RootFnOutput, RecurrentFnOutput

# The program takes the startPos variable with the coordinates in the first two indices and returns the next best move and probability distribution

#ACTION: 0->DOWN, 1->UP, 2->RIGHT, 3->LEFT

#(0, 0) AT TOP LEFT. (4, 4) AT BOTTOM 
#COORDINATES ARE (y, x), since its row x col

useRandomMaze = False
maze = jnp.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
])

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
    global downEmbeding

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





if __name__ == "__main__":
    if(useRandomMaze):
        maze = generateMaze()    
        while(not is_solvable(maze=maze)):
            maze = generateMaze()
        for row in maze:
            print(row)


# Run MCTS search for 0, 0

root = RootFnOutput(
    
    # Vector with important information, not exactly the raw state
    prior_logits=noisy_prior(jnp.array([[jnp.where(is_valid_tile(downEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(upEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(rightEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(leftEmbeding), 0.0, -jnp.inf)]]), rng_key),  # Estimated Policy
    value=jnp.array([0.0]),                      # shape (1,)
    embedding=startPos #jnp.zeros((1, 3))                  
)

policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations=100)
#path.append(policy_output.action.item())

def moveNextAction(policy_output, startPos, path):
    jax.debug.print("Next best move is: {}", policy_output.action.item())
    greedy  = int(jnp.argmax(policy_output.action_weights))
    jax.debug.print("Next best greedy move is: {}", greedy)
    print("Action probabilities:", policy_output.action_weights)
    
    action = greedy #policy_output.action.item()

    if action == 0:
        path.append("Down")
        startPos = startPos.at[0, 0].add(1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 0].add(1)
        startPos = startPos.at[0, 0].add(-1)
    elif action == 1:
        path.append("Up")
        startPos = startPos.at[0, 0].add(-1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 0].add(-1)
        startPos = startPos.at[0, 0].add(1)
    elif action == 2:
        path.append("Right")
        startPos = startPos.at[0, 1].add(1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 1].add(1)
        startPos = startPos.at[0, 1].add(-1)
    elif action == 3:
        path.append("Left")
        startPos = startPos.at[0, 1].add(-1)
        if(is_valid_tile(startPos)):
            startPos = startPos.at[0, 1].add(-1)
        startPos = startPos.at[0, 1].add(1)

    jax.debug.print("Position: {}", startPos)
    return startPos



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

    policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations=100) # Update invalid actions
    
print("Path: ", path) 
#print("Action probabilities:", policy_output.action_weights)