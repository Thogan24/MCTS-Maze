import jax
import jax.numpy as jnp
import mctx
from mctx import RootFnOutput, RecurrentFnOutput

root = RootFnOutput(
    prior_logits=jnp.array([[0.2, 5.1, 0.1]]),  # Estimated Policy
    value=jnp.array([0.0]),                      # shape (1,)
    embedding=jnp.zeros((1, 5))                  # Vector with important information, not exactly the raw state
)

# Next state predictor function
def recurrent_fn(params, rng_key, action, embedding):
    new_embedding = embedding + jnp.expand_dims(action, -1)
    
    # Reward: immediate payoff for taking this action
    reward = jnp.where(action == 0, 1.0, -1.0)
    # Flatten to ensure shape (1,) not (1,1) 
    reward = reward.flatten()
    
    discount = jnp.array([0.9])
    
    
    prior_logits = jnp.array([[0.1, 0.1, 0.1]])
    
    
    value = jnp.where(action == 0, 10.0, -1.0)
    value = value.flatten()
    
    output = RecurrentFnOutput(
        reward=reward,      
        discount=discount,  
        prior_logits=prior_logits,  
        value=value        
    )
    return output, new_embedding

params = None  # no params for dummy function
rng_key = jax.random.PRNGKey(0)

# Run MCTS search 
policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations=200)

print("Selected action:", policy_output.action)
print("Action probabilities:", policy_output.action_weights)