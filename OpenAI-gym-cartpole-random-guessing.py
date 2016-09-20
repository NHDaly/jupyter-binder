
# coding: utf-8

# ## Random Guessing Algorithm for CartPole OpenAI Gym
# This file implements the simplest Reinforcement Learning solution to the intro-to-RL CartPole problem from the OpenAI Gym (https://openai.com/requests-for-research/#cartpole), the "*random guessing algorithm*".

# In[1]:

from __future__ import print_function
import gym
import numpy as np


# In[2]:

env = gym.make('CartPole-v0')
print("Highs:", env.observation_space.high)
print("Lows: ", env.observation_space.low)

print(env.action_space)


# In[3]:

observation = env.reset()
observation


# In[4]:

# Here I'm just keeping track of my personal best. This has to be updated manually.
# ... I happened to randomly get a perfect model in the first 100 I tried (number 63)!
# They seem to happen in about 1/150 of the models I generate.
personal_best_reward = 1000
personal_best_weight = np.array([ 0.10047517,  0.45675998,  0.99510988,  0.75130867])


# In[5]:

def random_range(a,b,shape):
    return (b-a) * np.random.random(shape) + a


# In[6]:

# Create a bunch of random weights vectors, of the same shape as the input.
# That way, you can dot-product the input and the weight to make a choice.
Ws = [random_range(-1,1,env.observation_space.shape) for _ in range(100)]
Ws[:5]


# In[7]:

def model_step(W,x):
    ''' Simplest model ever: Just the linear multiplication, i.e. dot product! '''
    y = np.dot(W,x)
    return [0,1][y >= 0] # Use sign of result to decide left or right.
model_step(Ws[0],observation)


# In[8]:

def test_model(W):
    total_reward = 0
    num_batches = 100
    for i_episode in range(num_batches):
        observation = env.reset()
        done = False
        batch_reward = 0
        for _ in range(1000):
            #env.render()
            action = model_step(W,observation)
            observation, reward, done, info = env.step(action)
            batch_reward += reward
            if done:
                break
        #print("Batch Reward: {}".format(batch_reward))
        total_reward += batch_reward
    average_reward = total_reward/num_batches
    return total_reward,average_reward

test_model(Ws[2])


# In[9]:

best_weights = None
best_weights_idx = 0
best_weight_reward = 0


# In[10]:

for idx,W in enumerate(Ws):
    total_reward,average_reward = test_model(W)
    print("{0}/{1}: Average Reward: {2} Total Reward: {3}".format(idx, len(Ws), average_reward, total_reward))
    if average_reward > best_weight_reward:
        best_weight_reward = average_reward
        best_weights = W
        best_weights_idx = idx
print("Best Reward:", best_weight_reward)
print("Best Weight:", best_weights_idx)

if best_weight_reward > personal_best_reward:
    print("It's a NEW LAP RECORD!: {0}".format(best_weight_reward))
    print(best_weights)


# In[11]:

Ws[best_weights_idx]


# In[12]:

def render_model(W):
    total_reward = 0
    num_batches = 5
    for i_episode in range(num_batches):
        observation = env.reset()
        done = False
        batch_reward = 0
        print("{0}/{1}:".format(i_episode, num_batches))
        for _ in range(1000):
            #env.render()  # I don't think you can get this to render from MyBinder. :(
            action = model_step(W,observation)
            observation, reward, done, info = env.step(action)
            batch_reward += reward
            if done:
                break
        print("{0}/{1}: Batch Reward: {2}".format(i_episode, num_batches, batch_reward))
        total_reward += batch_reward
    average_reward = total_reward/num_batches
    return total_reward,average_reward


# In[13]:

render_model(Ws[best_weights_idx])


# In[14]:

render_model(personal_best_weight)

