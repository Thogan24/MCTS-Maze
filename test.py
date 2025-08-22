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
    x = jnp.array(state[0], int)
    y = jnp.array(state[1], int)
    in_bounds = jnp.logical_and(
        jnp.logical_and(0 <= x, x < maze.shape[1]),
        jnp.logical_and(0 <= y, y < maze.shape[0])
    )
    free_cell = maze[maze.shape[0] - y - 1, x] == 0
    return jnp.logical_and(in_bounds, free_cell)
    # if (0 <= x < maze.shape[1]) and (0 <= y < maze.shape[0]) and maze[maze.shape[0] - y - 1, x] == 0:
    #     return True
    # return False


state = [0, 0]
action = 0

state = jnp.array(state) # In case not in jnp
move = actions[action]
nextState = state + move

nextState = jnp.where(isValid(nextState), nextState, state)
nextState = jnp.array(nextState).reshape(-1)



reward = -jnp.abs(goal[0] - nextState[0]) + jnp.abs(goal[1] - nextState[1])/5.0
reward = jnp.where(jnp.all(nextState == goal), reward + 100, reward) # If statement but in jnp
reward = jnp.array([reward]) # Currently a scalar value, must switch to (1,)

print("nextState shape:", nextState.shape)
print("goal shape:", goal.shape)
print("raw reward shape:", reward.shape)

prior_logits = jnp.zeros((1, len(actions))) # Still uniform, THESE SHAPES NEED TO BE FIXED ALWAYS
value = -jnp.abs(goal[0] - nextState[0]) + jnp.abs(goal[1] - nextState[1]) 
value = jnp.array([value]) # Currently a scalar value, must switch to (1,)
