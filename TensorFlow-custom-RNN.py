
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import math
import random


# Load the Data!
# ------------
# Shakespeare's comedies!

# In[2]:

# data I/O
data = open('all_comedies_cat.txt', 'r').read() # should be simple plain text file
#data = "abcdefghijkabcdefghijk"
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


# In[3]:

def make_hot_vec(size, t):
    ''' size: a list of dimensions, like is passed to np functions.
        t:    the index to set to 1 (the truth).
    '''
    xs = np.zeros(size) # encode in 1-of-k representation
    xs[0,t] = 1
    return xs
make_hot_vec([1,4], 1)


# In[4]:

# Hyperparameters -- These control the model's behavior. It's nice to group them so you can
# change them together.
hidden_size = 100
num_steps = 25
learning_rate = 0.01

# This is just for printing our progress.
num_runs_between_logging = 100


# ----------
# Define the Model.
# ------
# This is modelled after Karpathy's minimal RNN, described in his [excellent article](karpathy.github.io/2015/05/21/rnn-effectiveness/) and implemented in this [short gist](https://gist.github.com/karpathy/d4dee566867f8291f086). The basic step function is:
# 
# ```
# # update the hidden state
# self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
# # compute the output vector
# y = np.dot(self.W_hy, self.h)
# ```
# 
# There are also two biases, `bh` and `by`. So the model has 5 learned parameters: `Wxh`, `Whh`, `Why`, `bh`, and `by`.

# In[5]:

# Define a single step in the RNN from one char to the next
Wxh = tf.Variable(tf.random_uniform([vocab_size, hidden_size], maxval=0.01)) 
Whh = tf.Variable(tf.random_uniform([hidden_size, hidden_size], maxval=0.01))
bh = tf.Variable(tf.zeros([hidden_size]))
Why = tf.Variable(tf.random_uniform([hidden_size, vocab_size], maxval=0.01)) 
by = tf.Variable(tf.zeros([vocab_size])) 


# ### Set up the recurrence
# 
# This is important: A recurrent neural net is *recurrent* because the simple model is repeated multiple times to create the overall model. This is called "unrolling" the net.
# 
# This matters, because in order for backpropogation to calculate the effect that the hidden state has on the loss, it needs to take the changes from one `state` to another into account when calculating the gradient. If you didn't unroll the network, and instead just returned the `state` from each run and passed the `state` back into the next run, the `state`'s only effect on the cost would be how the `state` effected `y`, not how the current `state` effected the next state.
# 
# For example, the simple network without unrolling it looks like this:
#   
# $  cost = truth - y $, where $ y = W_{hy}*h' $ and $h' = tanh(W_{hh}*h + W_{xh}*x) $
# 
# The current state, $h$, only effects $cost$ through it's impact on $y$. Even though it sets the *next state*, $h'$, the transition from $h$ to $h'$ is never considered during backpropogation.
# 
# Instead, an *unrolled* network *does* effect the cost both from the current $h$ and on its effect on the next $h$:
# 
# $  cost = (truth_1 - y_1) + (truth_0 - y_0) $
# 
# for $ y_1 = W_{hy}*h_1 $, $h_1 = tanh(W_{hh}*h_0 + W_{xh}*x_1) $
# 
# and
# $ y_0 = W_{hy}*h_0 $, $h_0 = tanh(W_{hh}*h + W_{xh}*x_0) $
# 
# Now, $W_{hh}$'s impact on the next $h$ effects the cost just as much as its impact on the current $y$.
# 
# SO, I *think*, the more steps you unroll a network for, the more emphasis you're placing on the hidden state's impact versus the other weights.

# In[6]:

inputs = [tf.placeholder(tf.float32, [None, vocab_size]) for _ in xrange(num_steps)]
outputs = {}
hs = {}
hs[-1] = tf.placeholder(tf.float32, [None, hidden_size])
for i in range(len(inputs)):
    hs[i]      = tf.nn.tanh(tf.matmul(inputs[i], Wxh) + tf.matmul(hs[i-1], Whh) + bh)
    outputs[i] = tf.nn.softmax(tf.matmul(hs[i], Why) + by)
print inputs[0], inputs[1], hs[-1], hs[0]


# In[7]:

sess = tf.Session()
# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()
# Run the 'init' op
sess.run(init_op)


# -----
# Training
# -------
# 
# This is the "cross-entropy" algorithm.
# $$
# H_{y'}(y)= −\sum_i y_i' \log(y_i)
# $$
#  $-\log(y_i)$ is useful because it turns percentages into (0,$\infty$) "cost".
# So then I think multiplying by the truth cancels out the percentages for everything else, and summing just turns the one_hot vector into a value?
# 
# For example:

# In[8]:

# Implement Cross Entropy:
truths = [tf.placeholder(tf.float32, [None, vocab_size]) for _ in xrange(num_steps)]
cross_entropies = [-tf.reduce_sum(truths[i]*tf.log(outputs[i])) for i in xrange(num_steps)]  # These operations act element-wise.
cross_entropy = tf.add_n(cross_entropies)


# In[9]:

# This was big. I changed the optimizer to the AdagradOptimizer, and upped the learning
#  rate from 0.01 to 0.1.
# I think it's okay to increase the learning rate because Adagrad decreases the rate over time.
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)


# # Do the thing!

# In[10]:

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# ## Hallucinating! The fun part!

# In[11]:

def hallucinate(sess, seed_ix, num_chars):
    ix = seed_ix
    ixes = [ix]
    hallucination_h_state = np.zeros([1, hidden_size]) 
    for i in range(num_chars):
        # Create a random starting letter 
        x_in = make_hot_vec([1,vocab_size], ix)

        feed_dict={inputs[0]: x_in}
        feed_dict.update({hs[-1]:hallucination_h_state})

        output, hallucination_h_state = sess.run((outputs[0],hs[0]), feed_dict=feed_dict)
        probs = output[0]
        ix = np.random.choice(range(len(probs)), p=probs)
        ixes.append(ix)
    hallucination=''.join([ix_to_char[ix] for ix in ixes])
    print hallucination


# ## Run it:

# In[12]:

losses = []
iterations = 0
ix = 0
h_state = np.zeros([1, hidden_size])
smooth_loss = -np.log(1.0/vocab_size)*num_steps # loss at iteration 0


# In[ ]:




# In[ ]:

def RunModel():
    global ix,losses,iterations,h_state,smooth_loss
    run = 0
    while True:
        letters=[]
        if ix + num_steps >= len(data):
            break
        input_chars  = [make_hot_vec([1,vocab_size], char_to_ix[ch]) for ch in data[ix:ix+num_steps]]
        target_chars = [make_hot_vec([1,vocab_size], char_to_ix[ch]) for ch in data[ix+1:ix+1+num_steps]]

        feed_dict={inputs[i]: input_chars[i] for i in range(len(input_chars))}
        feed_dict.update({truths[i]: target_chars[i] for i in range(len(target_chars))})
        feed_dict.update({hs[-1]:h_state})

        _, h_state, loss_out = sess.run((train_step, hs[num_steps-1], cross_entropy), feed_dict=feed_dict)

        smooth_loss = smooth_loss * 0.999 + loss_out * 0.001
        # Iterate by 1 or num_steps? I guess by 1 could lead to overfitting?
        ix += num_steps
        #        letters.append(data[ix])
        run += 1
        if run % num_runs_between_logging == 0:   
            iterations += 1
            #        print "train: ", ''.join(letters)
            print "========== Iteration ", iterations, " Loss: ", smooth_loss, " chars ", str(ix)+"/"+str(len(data)), " =============="
            losses.append(smooth_loss)
            hallucinate(sess, random.randint(0,vocab_size-1), 100)
            
RunModel()


# In[ ]:

hallucinate(sess, random.randint(0,vocab_size-1), 1000)


# In[ ]:

# RunModel()


# In[ ]:



