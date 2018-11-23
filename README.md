# DeepRL-Nanodegree-Project1 (Navigation)
Udacity Deep Reinforcement Nanodegree - Project 1 (Navigation)

In this project, we will train a Deep Q-Network (DQN) agent to try and solve Unity's Banana Collector environment.

### Environment Description

<img src="images/trained_agent.gif" width="50%" align="top-left" alt="" title="Banana Collector" />

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.
The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.

### Download Instructions

Here are the instructions to follow if you'd like to try out this agent on your machine. First, you'll need at least Python 3.6 installed on your system. You will also need these libraries to help run the code. Most of these can be installed using the 'pip install' command on your terminal once Python has been installed.

1. numpy - NumPy is the fundamental package for scientific computing with Python
1. collections - High-performance container datatypes
1. torch - PyTorch is an optimized tensor library for deep learning using GPUs and CPUs
1. unityagents - Unity Machine Learning Agents allows researchers and developers to transform games and simulations created using the Unity                  Editor into environments where intelligent agents can be trained using reinforcement learning, evolutionary strategies, or                  other machine learning methods through a simple to use Python API

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.
