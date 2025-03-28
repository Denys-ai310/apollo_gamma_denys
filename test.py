import gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Initialize Q-table with zeros
# Discretize the continuous state space for simplicity
def discretize_state(state):
    # Define bounds for each state dimension
    bounds = list(zip([-2.4, -2.0, -0.42, -2.0], [2.4, 2.0, 0.42, 2.0]))
    # Number of buckets per dimension
    n_buckets = (1, 1, 6, 3)
    
    ratios = [(state[i] + abs(bounds[i][0])) / (bounds[i][1] - bounds[i][0]) for i in range(len(state))]
    discrete_state = [min(n_buckets[i] - 1, int(ratio * n_buckets[i])) for i, ratio in enumerate(ratios)]
    return tuple(discrete_state)

# Initialize Q-table
n_states = (1, 1, 6, 3)  # Number of discrete states per dimension
n_actions = env.action_space.n  # Number of possible actions
Q = np.zeros(n_states + (n_actions,))

# Training parameters
n_episodes = 3000
max_steps = 200
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # For newer gym versions
    
    discrete_state = discretize_state(state)
    done = False
    total_reward = 0
    
    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[discrete_state])  # Exploit
        
        # Take action and observe next state and reward
        next_state, reward, done, info, _ = env.step(action)
        discrete_next_state = discretize_state(next_state)
        total_reward += reward
        
        # Update Q-value
        old_value = Q[discrete_state + (action,)]
        next_max = np.max(Q[discrete_next_state])
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        Q[discrete_state + (action,)] = new_value
        
        discrete_state = discrete_next_state
        
        if done:
            break
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Print episode statistics
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()

# Test the trained agent
env = gym.make('CartPole-v1', render_mode='human')
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    discrete_state = discretize_state(state)
    action = np.argmax(Q[discrete_state])
    state, reward, done, info, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"Test episode total reward: {total_reward}")
env.close()