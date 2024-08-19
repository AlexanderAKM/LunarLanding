# Lunar Landing Reinforcement Learning Solutions

This repository features implementations of four reinforcement learning (RL) algorithms applied to the classic Lunar Landing task from OpenAI Gym: Q-learning, SARSA, Deep Q-Network (DQN), and Double DQN. 

## Overview

The Lunar Landing task is a benchmark challenge in reinforcement learning where the agent must learn to control a lunar module safely onto the moon's surface. The project showcases:

- **Q-learning:** An off-policy algorithm for learning the optimal action-value function.
- **SARSA:** An on-policy algorithm that updates Q-values using the action performed by the current policy.
- **Deep Q-Network (DQN):** Uses deep neural networks to approximate the Q-value function, with techniques like experience replay and fixed Q-targets for stability.

<!--
## Getting Started

### Installation

First, clone this repository:
https://github.com/AlexanderAKM/Cartpole.git

Then, install the required dependencies:

```sh
pip install -r requirements.txt
```
### Running Agents

If you want to run an agent, for example, SARSA for 4000 episodes, use:
```sh
python src/main.py --agent sarsa --episodes 4000
```

### Plotting

If you want to plot the data from an experiment, for example, SARSA for 4000 episodes, use:
```sh
python src/plot.py -f data/input/rewards_sarsa_4000.csv
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* OpenAI Gym for providing the CartPole environment.
* Documentation of Pytorch on DQN implementation for cartpole: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Future Ideas

* Make a very, very simple MLP
* Do much more hyperparameter tuning -->