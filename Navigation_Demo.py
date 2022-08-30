from unityagents import UnityEnvironment
import numpy as np
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from agent import DemoAgent

import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="/Users/nathan/Documents/Learning/Udacity_deepRL/Value-based-methods/p1_navigation/Banana.app")

agent = DemoAgent(37, 4, 0, 'checkpoint_magic-violet-19.pth')

brain_name = env.brain_names[0] 
brain = env.brains[brain_name] 
env_info = env.reset(train_mode=True)[brain_name] # Train mode is fast moving

num_episodes = 500 
max_timesteps = 500 
scores = []
scores_window = deque(maxlen=100)

for episode_num in range(1, num_episodes+1):
    score = 0
    # reset environment 
    env_info = env.reset(train_mode=True)[brain_name]
    
    for timestep in range(max_timesteps):
        # Choose an action based on an inference 
        state = env_info.vector_observations[0]
        action = agent.act(state) 

        # Apply action to the environment and get the results 
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                   

        state = next_state
        score += reward      
        if done:
            break    
    
    scores_window.append(score)  # Save the most recent score 
    scores.append(score) 
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_num, np.mean(scores_window)), end="")
    if episode_num % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_num, np.mean(scores_window)))

env.close() 