import jax
import jax.numpy as jnp
import mctx
from mctx import RootFnOutput, RecurrentFnOutput
# Print entire path
# Fix no mcts version
# Maps with multiple routes
# Randomize maps


# The program takes the startPos variable with the coordinates in the first two indices and returns the next best move and probability distribution

#ACTION: 0->DOWN, 1->UP, 2->RIGHT, 3->LEFT

#(0, 0) AT TOP LEFT. (4, 4) AT BOTTOM 
#COORDINATES ARE (y, x), since its row x col
maze = jnp.array([
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

goal = [4, 4]

def is_valid_tile(embedding):
    row = embedding[0, 0].astype(int)
    col = embedding[0, 1].astype(int)
    in_bounds = (row >= 0) & (row < 5) & (col >= 0) & (col < 5)
    not_wall = jnp.where(in_bounds, maze[row, col] == 0, False)
    return not_wall & in_bounds
    

startPos = jnp.array([[0, 4, 0]])
downEmbeding = startPos.at[0, 0].add(1)
upEmbeding = startPos.at[0, 0].add(-1)
rightEmbeding = startPos.at[0, 1].add(1)
leftEmbeding = startPos.at[0, 1].add(-1)
rng_key = jax.random.PRNGKey(0)

def noisy_prior(prior_logits, rng_key, alpha=0.3, frac=0.25): 
    noise = jax.random.dirichlet(rng_key, jnp.full(prior_logits.shape[-1], alpha))
    p = jax.nn.softmax(prior_logits)
    mixed = (1 - frac) * p + frac * noise
    return jnp.log(mixed)

root = RootFnOutput(
    
    # Vector with important information, not exactly the raw state

    prior_logits=noisy_prior(jnp.array([[jnp.where(is_valid_tile(downEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(upEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(rightEmbeding), 0.0, -jnp.inf), jnp.where(is_valid_tile(leftEmbeding), 0.0, -jnp.inf)]]), rng_key),  # Estimated Policy
    value=jnp.array([0.0]),                      # shape (1,)
    embedding=startPos #jnp.zeros((1, 3))                  
)



# Next state predictor function
def recurrent_fn(params, rng_key, action, embedding):
    new_embedding = embedding
    new_embedding = jnp.where(action == 0, embedding.at[0, 0].add(1), new_embedding) # Down
    new_embedding = jnp.where(action == 1, embedding.at[0, 0].add(-1), new_embedding) # Up
    new_embedding = jnp.where(action == 2, embedding.at[0, 1].add(1), new_embedding) # Right
    new_embedding = jnp.where(action == 3, embedding.at[0, 1].add(-1), new_embedding) # Left
    is_terminal = (new_embedding[0,0].astype(int) == goal[0]) & (new_embedding[0,1].astype(int) == goal[1])
    valid = is_valid_tile(new_embedding)
    new_embedding = jnp.where(valid, new_embedding, embedding)
    new_embedding = jnp.where(is_terminal, new_embedding.at[0, 2].set(1), new_embedding) # Left

    jax.debug.print("embedding = {}", new_embedding)

    terminated = new_embedding[0, 2].astype(int)

    reward = jnp.where(terminated == 1, 1000, -((goal[0] - new_embedding[0, 0].astype(int)) + (goal[1]- new_embedding[0, 1].astype(int)))/5.0)
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

# Run MCTS search 
policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations=1000)

print("Selected action:", policy_output.action)
print("Action probabilities:", policy_output.action_weights)