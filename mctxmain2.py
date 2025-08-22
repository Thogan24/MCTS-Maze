import jax
import jax.numpy as jnp
import mctx
from mctx import RootFnOutput, RecurrentFnOutput

# Actions: (drow, dcol)
ACTIONS = jnp.array([
    [ 1,  0],  # 0 = Down
    [-1,  0],  # 1 = Up
    [ 0,  1],  # 2 = Right
    [ 0, -1],  # 3 = Left
], dtype=jnp.int32)

# --- Environment -------------------------------------------------------------

maze = jnp.array([
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=jnp.int32)

goal = jnp.array([4, 4], dtype=jnp.int32)  # (row, col)

def pos_is_valid(pos_rc: jnp.ndarray) -> jnp.ndarray:
    """pos_rc: shape (2,) -> True/False"""
    r, c = pos_rc[0], pos_rc[1]
    H, W = maze.shape
    in_bounds = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    return jnp.where(in_bounds, maze[r, c] == 0, False)

def prior_logits_for_state(embedding: jnp.ndarray) -> jnp.ndarray:
    """Return (1, 4) logits with -inf for invalid actions from this state."""
    # embedding shape (1, 3): [row, col, done_flag]
    pos = embedding[0, :2].astype(jnp.int32)                 # (2,)
    candidates = pos[None, :] + ACTIONS                      # (4, 2)
    valid_mask = jax.vmap(pos_is_valid)(candidates)          # (4,)
    logits = jnp.where(valid_mask, 0.0, -jnp.inf)            # (4,)
    return logits[None, :]                                   # (1,4)

# --- Root --------------------------------------------------------------------

# Embedding: [row, col, done_flag]
root = RootFnOutput(
    prior_logits=jnp.array([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
    value=jnp.array([0.0], dtype=jnp.float32),
    embedding=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),  # start at (0,0), not done
)

# --- Model (recurrent) -------------------------------------------------------

def recurrent_fn(params, rng_key, action, embedding):
    """
    embedding shape: (1, 3) -> [row, col, done_flag]
    action: scalar int
    Returns: (RecurrentFnOutput, next_embedding)
    """
    # Current position & done flag
    pos = embedding[0, :2].astype(jnp.int32)           # (2,)
    done_flag = embedding[0, 2]                        # scalar (0.0 or 1.0)

    # If already terminal, stay in place and prevent further expansion.
    already_terminal = (done_flag == 1.0)

    # Proposed move
    delta = ACTIONS[action]                            # (2,)
    proposed_pos = pos + delta                         # (2,)
    valid = pos_is_valid(proposed_pos)

    # Apply move only if valid and not already terminal
    new_pos = jnp.where((~already_terminal) & valid, proposed_pos, pos)

    # Check if goal reached now (or was already done)
    now_terminal = already_terminal | jnp.all(new_pos == goal)

    # Build next embedding: [row, col, done_flag]
    next_embedding = jnp.array([[new_pos[0], new_pos[1], jnp.where(now_terminal, 1.0, 0.0)]],
                               dtype=jnp.float32)

    # --- Rewards & Values ---
    # Manhattan distance to goal (negative = better when smaller magnitude)
    dist = (goal[0] - next_embedding[0, 0].astype(jnp.int32)) + \
           (goal[1] - next_embedding[0, 1].astype(jnp.int32))
    # Base reward encourages moving toward goal
    base_reward = -dist.astype(jnp.float32) / 5.0

    # Penalize invalid/self-loop moves (no position change and not terminal)
    no_move = jnp.all(new_pos == pos)
    penalty = jnp.where((~now_terminal) & no_move, -1.0, 0.0)

    # Give big terminal bonus
    terminal_bonus = jnp.where(now_terminal, 1000.0, 0.0)

    reward = (base_reward + penalty + terminal_bonus)[None]   # shape (1,)

    # Discount stops at terminal
    discount = jnp.where(now_terminal, jnp.array([0.0], dtype=jnp.float32),
                                      jnp.array([0.99], dtype=jnp.float32))

    # Prior logits for the *reached* node: mask invalid actions there.
    # If terminal, set all to -inf to prevent expansion.
    next_priors = jnp.where(
        now_terminal,
        jnp.full((1, 4), -jnp.inf, dtype=jnp.float32),
        prior_logits_for_state(next_embedding)
    )

    # Simple value = -distance (encourages closeness to goal)
    value = (-dist.astype(jnp.float32))[None]  # shape (1,)

    out = RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=next_priors,
        value=value,
    )
    return out, next_embedding

# --- Run MCTS ---------------------------------------------------------------

params = None
rng_key = jax.random.PRNGKey(0)

policy_output = mctx.gumbel_muzero_policy(
    params, rng_key, root, recurrent_fn, num_simulations=1000
)

print("Selected action:", policy_output.action)
print("Action probabilities:", policy_output.action_weights)
