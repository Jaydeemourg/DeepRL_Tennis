## 1. Project Goal

In this project, I developed a reinforcement learning (RL) agent that controls a robotic arm using Unity's [tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. The goal is to get each agent to keep the ball in play

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## 2. Solution Approach
Here are the steps taken in developing an agent that solves the environment.

* Identify the state and action space.
* Select ddpg algorithm and implementing it.
* Train the agent

#### Identify the state and action space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

#### Select ddpg algorithm and implement it
the action space of the task is continuous i.e. there's an unlimited range of possible action values to control. DDPG is suitable for this type of action space

**DDPG algorithm (Deep Deterministic Policy Gradient):**
this algorithm consists of two networks;
- a Critic that measures how good the action taken is using value-based method. The value function maps each state action pair to a value which quantifies how it is. The value function calculates what is the maximum expected future reward given a state and an action.

- an Actor that controls how the agent behaves using policy-based method.  The policy is optimized without using a value function. This is useful when the action space is continuous or stochastic.

an update is made at each step using TD Learning, this is done without waiting until the end of the episode. The Critic observes the agent's action and provides feedback in order to update the policy and be better at playing that game.
    
    
**Experience Replay**
Experience replay allows the RL agent to learn from past experience.

DDPG also utilizes a replay buffer to gather experiences from each agent. Each experience is stored in a replay buffer as the agent interacts with the environment. 

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). Each agent samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

#### Network architecture

Two deep neural networks comprising of Actor Critic models are used;

The Actor Network receives as input 8 variables representing the state size, with two hidden layers each with 128 and 128 nodes. I used ReLU activation functions on the hidden layers and tanh on the output layers. That means, the Actor is used to approximate the optimal policy π deterministically.

The Critic Network receives as input 8 variables representing the observation space , also with two hidden layers each with 128 and 128 nodes.
The output of this network is the prediction of the target value based on the given state and the estimated best action.
That means the Critic calculates the optimal action-value function Q(s, a) by using the Actor's best-believed action.

**Hyperparameters** 

 Parameters | Value | Description
----------- | ----- | -----------
BUFFER_SIZE | int(1e5) | replay buffer size
BATCH_SIZE | 128 | minibatch size
GAMMA | 0.99 | discount factor
TAU | 1e-3 | for soft update of target parameters
LR_ACTOR | 2e-4 | learning rate of the actor
LR_CRITIC | 2e-4 | learning rate of the critic
WEIGHT_DECAY | 0 | L2 weight decay

#### Train the agent 
The ddpg agent is then trained for 1500 episodes until the performance threshold is realized.  

## 3. Performance for DDPG Agent.
Environment was solved in 1362 episodes. Average score: 0.5184000077843666
![alt text](https://github.com/Jaydeemourg/DeepRL_Continuous_Control/blob/main/score_per_episode_plot.png)

## 4. Future Improvements 
- **Add prioritized experience replay**  Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.


**Experiment with other algorithms like**
- PPO [paper](https://arxiv.org/abs/1707.06347) 
- (A2C) [paper](https://arxiv.org/abs/1602.01783v2)
- D4PG
