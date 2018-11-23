# Description of the Implementation

## Learning Algorithm

The model used for training the agent was a simple 3 layer network with the following architecture:
- Input Layer: State-Size (which happens to be 37 in our case)
- First Hidden Layer: 64
- Second Hidden Layer: 64
- Output Layer: Action-Size (which in our case is 4)

To train our agent, I used a replay buffer/ replay memory to store the experiences that the agent has with the environment after every
timestep. The replay memory has a size of 10^5 which means that it can store 10^5 experience tuples (state, action, reward, next_state, done).

Once the replay memory has >= 64 experience tuples, at every fourth timestep, we sample a batch of 64 tuples to train with. The agent then
learns from these experiences and updates its Q-Network weights. Another thing that happens during these 4th timesteps is the copying of 
the local Q-Network's weights onto the target Q-Network. This is known as a soft update. This step is done in order to use the target weights
in the future timesteps as a label, to calculate and reduce the error between the target and local networks while training the agent.

## Hyperparameters used

1. BUFFER_SIZE = int(1e5)    # replay buffer size
1. BATCH_SIZE = 64           # mini-batch size
1. GAMMA = 0.99              # discount factor
1. TAU = 1e-3                # for soft update of target parameters
1. LR = 5e-4                 # learning rate
1. UPDATE_EVERY = 4          # how often to update the network
1. fc1_units = 64            # First hidden layer size
1. fc2_units = 64            # Second hidden layer size
1. epsilon_start = 1.0       # epsilon start for the epsilon-greedy policy
1. epsilon_end = 0.01        # epsilon end for the epsilon-greedy policy
1. epsilon_decay = 0.98      # epsilon decay rate

## Plot of Rewards

A plot of the training agent can be found below. As you can see, the more we train the agent, the higher the rewards it collects by updating the 
Deep Q-Network. In this plot, the agent was able to solve the episode and achieve a score of 13.0 in as few as 220 episodes!

<img src="images/Training Graph.png" align="top-left" alt="" title="Training Graph" />

Here's a plot of the trained agent's scores over 10 episodes:

<img src="images/Test Graph.png" align="top-left" alt="" title="Test Graph" />

Episode 1	Average Score: 19.00

Episode 2	Average Score: 18.00

Episode 3	Average Score: 14.00

Episode 4	Average Score: 14.75

Episode 5	Average Score: 16.60

Episode 6	Average Score: 17.00

Episode 7	Average Score: 15.00

Episode 8	Average Score: 15.12

Episode 9	Average Score: 15.22

Episode 10	Average Score: 15.10

## Future Ideas

In order to try and improve the network I'm going to try and implement the following algorithms that were briefly covered during the course:
1. Double DQN - Where we can address the overestimation of Q-Value problem by using two neural nets. One set of parameters w is used to
select the best action and another set of parameters w' is used to evaluate that action

1. Dueling DQN - Where we utilize two streams: one that estimates the state value function V(s) and another that estimates the advantage
for each action A(s,a). These two values are then combined to obtain the desired Q-values.

1. Prioritized Experience Replay - A framework where we replay important transitions more frequently, and therefore learn more efficiently.

1. *Maybe* Rainbow DQN :) - An algorithm that combines six extensions to the DQN algorithm viz. DDQN, Prioritized DDQN, Dueling DDQN, A3C,
Distributional DQN and Noisy DQN (A total of 7 algorithms)
