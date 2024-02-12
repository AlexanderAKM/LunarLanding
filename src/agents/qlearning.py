import numpy as np
import gymnasium as gym
import time
import math
import pandas as pd

def run(episodes):
    env = gym.make("LunarLander-v2", render_mode = "human")

    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    SHOW_EVERY = 1000

    n_bins = [10, 10, 10, 10, 10, 10, 2, 2]

    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.001
    #print(f"low observation state = {env.observation_space.low}")
    #print(f"high observation state = {env.observation_space.high}")

    q_table = np.random.uniform(low=0, high=1, size=(n_bins + [env.action_space.n]))

    def get_discrete_state(state, bins):
        # Observation space: [pos_x, pos_y, vel_x, vel_y, angle, angular_vel, left_leg_contact, right_leg_contact]
        #discretize them here.
        discretized = list()
        
        for i, val in enumerate(state):
            if i < 6:
                discretized.append(np.digitize(val, bins[i]) - 1)
            else:
                discretized.append(int(val))
        
        return tuple(discretized)

    bins = [
        np.linspace(-1.0, 1.0, n_bins[0]), # position of x coordinate
        np.linspace(-1.0, 1.0, n_bins[1]), # position of y coordinate
        np.linspace(-3.5, 3.5, n_bins[2]), # linear velocity of x
        np.linspace(-3.5, 3.5, n_bins[3]), # linear velocity of y
        np.linspace(-2.0, 2.0, n_bins[4]), # angle to the ground
        np.linspace(-3.5, 3.5, n_bins[5]), # angular velocity
    ]

    rewards_episodes = {'Episode': [], 'Reward': []}

    for episode in range(episodes + 1):
        state = env.reset()
        discrete_state = get_discrete_state(state[0], bins)
        total_reward = 0
        terminated, truncated = False, False
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-decay_rate * episode)

        if episode % SHOW_EVERY == 0:
            render = True
            print(episode)
        else:
            render = False

        while not terminated and not truncated:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state, bins)
            total_reward += reward

            if render:
                env.render()
                time.sleep(0.01)
            
            if not terminated and not truncated:
                #print(discrete_state)
                #print(action)
                current_q = q_table[discrete_state + (action,)]
                new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[new_discrete_state]) - current_q)
                q_table[discrete_state + (action,)] = new_q
            
            discrete_state = new_discrete_state

        rewards_episodes['Episode'].append(episode)
        rewards_episodes['Reward'].append(total_reward)
        
    rewards_file_path = f'data/input/rewards_qlearning_{episodes}.csv'
    df_rewards = pd.DataFrame(rewards_episodes)
    df_rewards.to_csv(rewards_file_path, index=False)

    env.close()

if __name__ == "__main__":
    run(episodes=2000)



