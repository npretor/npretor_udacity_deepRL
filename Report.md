1. Network design 
2. Hyperparameter explanations 
3. Goal 

## Learning algorithm 
I chose a DQN - a deep Q network. A deep Q network is a policy learner using a deep neural net. The network is essentially a policy estimator function mapping input states to output actions. 

My network design is comprised of fully connected network with relu between each layer. I experimented with different layer sizes and adding an additional layer, but there was no discernable improvement on the performance. 
The input space is the observed environment, it has 37 dimensions. 
The output space is the the possible actions. There are 4 actions that can be taken
- move forward 
- move backwards 
- move left 
- move right 



## Hyperparameters explanations 
### Learning rate
Step size that gradient descent uses. Too small and you take too long, too large and you overshoot and aren’t precise enough

### Batch size
Size of bundle of experiences to learn from passed to the GPU. Limited by GPU RAM. 

### Replay buffer size
How many action-steps to store before we learn

### Update frequency
How many 

### Gamma
Rate at which old rewards are discounted, esentially the rate at which the model forgets past experiences

### Tau: 
At a very basic level, we have to ask ourselves, how do we know we are moving towards our goal? Our goal is maximizing reward. Our goal is twofold in the case of the banana gathering challenge: (1) avoid blue bananas and (2) get yellow bananas. Each sub-goal has an associated reward, and we want to maximize overall reward. A deep Q network is learning the best action to take based on what it sees in front of it. Its inputs are the environment it can “see”, and its outputs are the action probabilities, with the highest being the decision chosen. When we start off, we randomly initialize the model’s weights and biases, then we have the agent take a few actions. After a few actions, we learn from the experiences and and update the model. This is where the Tau hyperparameter comes in. We update the models using a “soft update”(see paper https://arxiv.org/pdf/2008.10861.pdf). This is done by actually using two models. One model is our “main” model, which we learn from, and the other is the “target” model. Every time before we train on new experiences, we make a copy of existing model. We then train the “target” network on the new data. After training, we extrapolate between the weights and biases from the “target” network and the “main” network, but there is a tradeoff. Too much target updating and the model’s predictions become noisy ( and don’t settle towards the ideal case as fast), too little updating and we learn too slowly. 

## Goal
The environment is considered to be solved when over a period of a hundred episodes the agent averages a reward of 13 or more

