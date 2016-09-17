
# coding: utf-8

# ## Hill Climbing Algorithm for CartPole OpenAI Gym
# This file implements the simplest Reinforcement Learning solution to the intro-to-RL CartPole problem from the OpenAI Gym (https://openai.com/requests-for-research/#cartpole), the "*hill-climbing algorithm*".

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

# HyperParameters:
learning_rate = 1  # There is no reason to keep this small, since we're perturbing by a random amount within this range.
                   # Since this isn't gradient descent and we only keep ones that perform better, you don't have the
                   # "overshooting" problem of too large a learning_rate. Here I've set it to the whole space.
num_runs = 10000
num_batches = 100
max_num_steps = 1000


# In[6]:

def random_range(a,b,shape):
    return (b-a) * np.random.random(shape) + a


# In[7]:

W_initial = random_range(-1,1,env.observation_space.shape)
W_initial


# In[8]:

def model_step(W,x):
    ''' Simplest model ever: Just the linear multiplication, i.e. dot product!
    Technically this is a logistic regression, which chooses between two classes. '''
    y = np.dot(W,x)
    return [0,1][y >= 0] # Use sign of result to decide left or right.
model_step(W_initial,observation)


# In[9]:

def update_model(W_prev,score_prev, W, score):
    ''' Randomly perturb the weights, and if it performs better than last time, keep it. '''
    keep_W = W_prev
    keep_score = score_prev
    if score > score_prev:
        keep_W = W
        keep_score = score
        print("-- Replacing old model! {0} better than {1} --".format(score, score_prev))
    new_W = np.copy(keep_W)
    new_W += random_range(-learning_rate, learning_rate, new_W.shape)
    return new_W, keep_W, keep_score
Wb = random_range(-1,1,env.observation_space.shape)
new_W, prev_W, prev_score = update_model(W_initial,10, Wb, 12)
W_initial, Wb, new_W, prev_W, prev_score


# In[10]:

def max_possible_reward():
    return max_num_steps


# In[11]:

def test_model(W):
    total_reward = 0
    for i_episode in range(num_batches):
        observation = env.reset()
        done = False
        batch_reward = 0
        for _ in range(max_num_steps):
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

test_model(W_initial)


# In[12]:

prev_weights = W_initial
prev_reward = 0
W = np.copy(prev_weights)


# In[13]:

for idx in range(num_runs):
    global prev_weights,prev_reward,W
    total_reward,average_reward = test_model(W)
    print("{0}/{1}: Average Reward: {2} Total Reward: {3}".format(idx, num_runs, average_reward, total_reward))
    W, prev_weights, prev_reward = update_model(prev_weights, prev_reward, W, total_reward)
    
    if average_reward == max_possible_reward():
        break
    
best_weights,best_weight_reward = W,total_reward
print("Best Reward:", best_weight_reward)
print("Best Weight:", best_weights)

if best_weight_reward > personal_best_reward:
    print("It's a NEW LAP RECORD!: {0}".format(best_weight_reward))
    print(best_weights)


# In[14]:

best_weights


# In[15]:

def render_model(W):
    total_reward = 0
    num_batches = 5
    for i_episode in range(num_batches):
        observation = env.reset()
        done = False
        batch_reward = 0
        print("{0}/{1}:".format(i_episode, num_batches))
        for _ in range(5000):
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


# In[16]:

render_model(best_weights)


# In[17]:

render_model(personal_best_weight)


# In[ ]:



