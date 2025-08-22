import jax
import jax.numpy as jnp
import mctx
from mctx import RootFnOutput, RecurrentFnOutput
import functools

maze = jnp.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
])

goal = jnp.array([4, 4]) #?
actions = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]])



def isValid(state): # Checks if state is a wall or out of bounds
    x = state[0] # jnp.array(state[0], int)
    y = state[1] #jnp.array(state[1], int)
    in_bounds = jnp.logical_and(
        jnp.logical_and(0 <= x, x < maze.shape[1]),
        jnp.logical_and(0 <= y, y < maze.shape[0])
    )
    free_cell = maze[maze.shape[0] - y - 1, x] == 0
    return jnp.logical_and(in_bounds, free_cell)

def root(params, rng_key): # The initial state
    return mctx.RootFnOutput(
        prior_logits = jnp.zeros((1, len(actions))), # Uniform distribution, no assumptions at the start
        value=jnp.array([-8.0]),
        embedding=jnp.array([[0, 0]])  # start state
    )

def recurrent_fn(params, rng_key, action, state, step):
    state = jnp.array(state) # In case not in jnp
    state = jnp.array(state).reshape(-1)
    move = actions[action]
    nextState = state + move
    nextState = jnp.where(isValid(nextState), nextState, state)
    
    #nextState = jnp.array([nextState[0], nextState[1]])
    nextState = jnp.array(nextState).reshape(-1)

    reward = -jnp.abs(goal[0] - nextState[0]) + jnp.abs(goal[1] - nextState[1])/5.0
    reward = jnp.where(jnp.all(nextState == goal), reward + 100, reward) # If statement but in jnp
    reward = jnp.array([reward]) # Currently a scalar value, must switch to (1,)

    discount = jnp.array([0.99])      # shape (1,)
    #reward = reward.reshape(1)         # shape (1,)
    reward=reward[None]


    prior_logits = jnp.zeros((1, len(actions))) # Still uniform, THESE SHAPES NEED TO BE FIXED ALWAYS
    value = -jnp.abs(goal[0] - nextState[0]) + jnp.abs(goal[1] - nextState[1]) 
    value = jnp.array([value]) # Currently a scalar value, must switch to (1,)
    
    print("nextState shape:", nextState.shape)
    print("goal shape:", goal.shape)
    print("raw reward shape:", reward.shape)
    output = mctx.RecurrentFnOutput(
        reward=reward,                 # shape (1,)
        discount=discount,     # shape (1,)
        prior_logits=prior_logits, # shape (1, num_actions)
        value=value,                   # shape (1,)
    )
    return output, nextState


rng = jax.random.PRNGKey(0)
output = mctx.muzero_policy(
    params=None,
    rng_key=rng,
    root=root(None, rng),
    recurrent_fn=functools.partial(recurrent_fn, None),
    num_simulations=3000,
    max_depth=40,
)