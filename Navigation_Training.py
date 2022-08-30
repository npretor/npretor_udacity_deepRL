# !python3 -m pip install wandb

from unityagents import UnityEnvironment
import agent
import numpy as np
import wandb
wandb.init(project="DeepRL_bananaFinder", entity="nextflex") 

env = UnityEnvironment(file_name="/Users/nathan/Documents/Learning/Udacity_deepRL/Value-based-methods/p1_navigation/Banana.app")

# get the default brain
brain_name = env.brain_names[0] 
brain = env.brains[brain_name] 

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)



agent = agent.Agent(37, 4, 0)

scores = []

num_episodes=1000 
max_timesteps=1000
epsilon_start=1.0
epsilon_end=0.01
epsilon_decay=0.995


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3                # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


wandb.init(config={
    "num_episodes": num_episodes,
    "max_timesteps": max_timesteps,
    "epsilon_start": epsilon_start,
    "epsilon_end": epsilon_end,
    "epsilon_decay": epsilon_decay,
    "buffer_size": BUFFER_SIZE, 
    "batch_size": BATCH_SIZE,
    "gamma": GAMMA,
    "tau": TAU,
    "learning_rate": LR,
    "update_every": UPDATE_EVERY 
})


from collections import deque
import torch
import matplotlib.pyplot as plt
%matplotlib inline

scores = []

def dqn(num_episodes, max_timestep, epsilon_start, epsilon_end, epsilon_decay):
    """
    max_timesteps: limits the number of timesteps so we don't get stuck in scenarios with really low rewards or loops
    
    1. Iterate through a set number of episodes
    2. For each episode, reset the environment and the score(reward sum)
    """

    #scores = []
    scores_window = deque(maxlen=100)
    epsilon = epsilon_start
    
    for episode_num in range(1, num_episodes+1):
        # Run through a set number of timesteps, getting rewards and training every so often. Decrease epsilon every episode 
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        #print(state)
        
        for timestep in range(max_timesteps):
            # Choose an action based on an inference 
            action = agent.act(state, epsilon) 

            # Apply action to the environment and get the results 
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]               
            
            # Add the result to memory and train every few preset timesteps 
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break
                
        scores_window.append(score)  # Save the most recent score 
        scores.append(score) 
        epsilon = max(epsilon_end, epsilon_decay*epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_num, np.mean(scores_window)), end="")
        if episode_num % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_num, np.mean(scores_window)))
            wandb.log({"average_score": np.mean(scores_window)}) 
        if np.mean(scores_window)>=15.0:
            # If average scores across the current window of 100 scores is 200 or greater, we are done 
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_num-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), './checkpoint.pth')
            break
    return scores


scores = dqn(num_episodes, max_timesteps, epsilon_start, epsilon_end, epsilon_decay)


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show() 

torch.save(agent.qnetwork_local.state_dict(), './checkpoint_{}.pth'.format(wandb.run.name)) 
wandb.finish()