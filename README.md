[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

### Environment Description
#### States
* The observation space consists of 33 dimensions.
* It contains position, rotation, velocity, and angular velocities of the arm. 

#### Actions
* Each action is a vector with four numbers. 
* The numbers are corresponding to torque applicable to two joints. 
* Every entry in the action vector is a number between -1 and 1.

#### Rewards
* A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
* +0 Otherwise 

#### Goal
* The goal of your agent is to maintain its position at the target location for as many time steps as possible.
* The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


###  Project Structure
The repository contains the following files.
* Continous_Control.ipynb Contains the agent training code for Unity Reacher environment.
* ddpg_agent.py Contains DDPG based agent implemenation.
* network.py Contains actor and critic network.
* noise.py Contians Ornstein-Uhlenbeck noise process utility class.
* replay_buffer.py Contains replay buffer utility class.


### Getting Started
1. Install Anaconda(https://conda.io/docs/user-guide/install/index.html)
2. Install dependencies by issue:
```
pip install -r requirements.txt
```
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Place the file in the root folder, and unzip (or decompress) the file.  

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training.  
