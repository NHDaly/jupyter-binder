
original_text = ""
open("all_comedies_cat.txt") do original_text_file
    global original_text
    original_text = readall(original_text_file)
end
# Print some samples from the text, just to get a feel for it.
println("Length: ", length(original_text))
println("\n --------- Some Sample Text: -------- \n")
println(original_text[1:200])
println("\n --------- ... ---------------------- \n")
print(original_text[end-1000:end])

alphabet = sort(unique(original_text))  # Converts an index into a character.
println(length(alphabet))

reverse_alphabet = [ch => i for (i,ch) in enumerate(alphabet)]  # Converts a char into an index.

"""
A Recurrent Neural Network is a collection of weight matrices, which are updated after each training batch,
and a "hidden state", h, which is updated after each classification in the batch and carried throughout the whole batch.
"""
type RNN
    # The Weights and Biases (matrices)
    Wxh   # Controls how each input neuron affects the new hidden neuron values.
    Whh   # Controls how each hidden neuron affects the new hidden neuron values.
    Why   # Controls how to calculate the output neurons from the hidden layer.
    bh    # Bias added to new hidden state.
    by    # Bias added to calculated output vector.

    # Adagrad update "memory" weights. This is an implementation detail of the Adagrad update algorithm.
    mWxh
    mWhh
    mWhy
    mbh
    mby
    
    # The hidden state is updated by each step, and is a part of generating the next step's classification. It is not
    # updated by backpropogation (training), but it's part of calculating the new update for the weights after each step.
    # Updating the hidden state and carrying it to the next step is called "unrolling" the RNN, and is what makes an
    # RNN "deep", similar to having many hidden layers in a more traditional RNN.
    h     # Not a variable, but still associated with the RNN.
    
    function RNN(in_dim::Integer, hidden_dim::Integer, out_dim::Integer)
        # Initialize the Weights/Biases variables
        Wxh = randn(hidden_dim, in_dim)*0.01      # scaled way down.
        Whh = randn(hidden_dim, hidden_dim)*0.01  # Initialize these with random values from normal distribution,
        Why = randn(out_dim, hidden_dim)*0.01
        bh = zeros(hidden_dim, 1)
        by = zeros(out_dim, 1)
        
        # Initialize the memory variables
        mWxh, mWhh, mWhy, mbh, mby = zeros(Wxh), zeros(Whh), zeros(Why), zeros(bh), zeros(by)

        # Initialize the hidden state vector.
        h = zeros(bh)
        
        new(Wxh, Whh, Why, bh, by, mWxh, mWhh, mWhy, mbh, mby, h)
    end
end
# Test code
r = RNN(1,2,1)
println(r.Wxh)
println(r.h)

"""
A helper function to prevent spiraling values when doing derivatives.
"""
function clip_values(deltas_matrix, max_threshold::Float64, min_threshold::Float64)
    deltas_matrix = max(deltas_matrix, min_threshold)
    deltas_matrix = min(deltas_matrix, max_threshold)
    return deltas_matrix
end
# Test code
clip_values([1.0 -11.0 6.0]', 5.0, -5.0)

"""
    forward_pass_step(r::RNN, in_h, x)

A single step in the RNN: calculate a new output vector and new hidden
state from an input vector (a one-hot vector) and the current hidden state.

Returns the output vector as a raw vector and as a normalized
vector, which can be thought of as a probability distribution, and the new hidden state.
"""
function forward_pass_step(r::RNN, in_h, x)
    const local Wxh = r.Wxh
    const local Whh = r.Whh
    const local Why = r.Why
    const local bh = r.bh
    const local by = r.by

    # Calculate the new h.
    const new_h = tanh(Wxh*x + Whh*in_h + bh)

    const y = Why*new_h + by
    
    # --- Normalize y so that it represents a probability distribution over the set of characters.
    
    # Taking exp(y) conveniently eliminates negative values from tanh.
    const exp_y = exp(y)     # 2x1 (pointwise)
    const y_norm = exp_y ./ sum(exp(y)) # 2x1 Element wise division 

    return y_norm, y, new_h
end

# Test code
r1 = RNN(2,4,3)
r1.h = [0 0 0 0.]'
y1, _, r1.h = forward_pass_step(r1, r1.h, [1.0 0]')
y2, _, r1.h = forward_pass_step(r1, r1.h, [0.0 0.1]')
y3, _, r1.h = forward_pass_step(r1, r1.h, [1.0 0.]')
println("ys:\n$y1\n$y2\n$y3")
println("h:\n",r1.h)

"""
    forward_pass_backpropogate_batch(r::RNN, xs, ts)

Unroll the RNN for all the inputs in this batch, then perform backpropogation and calculate the
delta updates for its parameters. Returns the `cost` calculated over this batch, and the delta updates.

This function modifies r.h, and leaves it in its new state.

r - an RNN.
x - The batch of input vectors: a tuple of one-hot vectors.
t - The batch of "truth" vectors: a tuple of one-hot vectors of same count as xs.
"""
function forward_pass_backpropogate_batch(r::RNN, xs, ts)

    #Forward Pass
    hs, ys, ps = Dict(), Dict(), Dict()
    hs[0] = r.h
    local Wxh = r.Wxh
    local Whh = r.Whh
    local Why = r.Why
    local bh = r.bh
    local by = r.by

    out_delta_max_threshold = 5.0   # Tweakable.
    out_delta_min_threshold = -5.0   # Tweakable.

    cost = 0
    
    # Unroll the network.
    # For each input, perform the forward pass step and update the hidden state. Keep track
    # of the state after each step so we can calculate the derivatives correctly during backpropogation.
    for i in 1:length(ts)
        local h = hs[i-1]
        local x = xs[i]
        local t = ts[i]

        ps[i], ys[i], hs[i] = forward_pass_step(r, h, x)

        # Take the dot-product with t-inverse, to get only
        # the value from log(ps[i]) which corresponds to the truth.
        prediction_for_truth_value = t' * ps[i]  # 1x1 (scalar-ish)

        # The log-cost reflects how close the prediction for the truth value was to 1.
        cost += -log(prediction_for_truth_value)
    end

    # ----------  Now go back down:
    # Starting from the cost, calculate the derivative (gradient) of the cost, with respect to each variable,
    # or, *how much the cost changes if you change each variable*. This allows us to move each variable along
    # it's derivative to lower the cost for this batch.

    # These will be our final outputs, they represent ``δcost/δvariable``.
    dWxh = zeros(r.Wxh)
    dWhh = zeros(r.Whh)
    dWhy = zeros(r.Why)
    dbh  = zeros(r.bh)
    dby  = zeros(r.by)
    
    ∂cost_∂hnext = zeros(r.h)
    for i in length(ts):-1:1

        # Copied from Karpathy. I don't know what these two lines mean.
        #dy = ps[i]
        #dy -= ts[i] # backprop into y
        #dWhy += dy * hs[i]'
        #dby += dy
        #dh = Why' * dy + dhnext # backprop into h
        #dhraw = (1 - hs[i] .* hs[i]) .* dh # backprop through tanh nonlinearity
        #dbh += dhraw
        #dWxh += dhraw * xs[i]'
        #dWhh += dhraw * hs[i-1]'
        #dhnext = Whh' * dhraw
        
        
        ∂cost_∂y = ps[i]
        ∂cost_∂y -= ts[i] # backprop into y
 
        ∂cost_∂Why = ∂cost_∂y * hs[i]'  # 2x3
        ∂cost_∂by = ∂cost_∂y            # 2x1
 
        ∂cost_∂h = Why' * ∂cost_∂y  +  ∂cost_∂hnext   # 3x1
        dhraw = (1 - hs[i] .^ 2) .* ∂cost_∂h    # "backprop through tanh nonlinearity"
 
        ∂cost_∂bh = dhraw  # 3x1    # noop
 
        ∂cost_∂Whh = dhraw * hs[i-1]'   # 3x3
        ∂cost_∂Wxh = dhraw * xs[i]'   # 3x2
 
        ∂cost_∂hnext = Whh' * dhraw  # 3x1
 
        # Update the final derivatives
 
        dWxh += ∂cost_∂Wxh
        dWhh += ∂cost_∂Whh
        dWhy += ∂cost_∂Why
        dbh  += ∂cost_∂bh
        dby  += ∂cost_∂by


        #println("∂cost_∂Wxh:$∂cost_∂Wxh")
    end

    # Clip deltas to mitigate exploding gradients.
    dWxh = clip_values(dWxh, out_delta_max_threshold, out_delta_min_threshold) 
    dWhh = clip_values(dWhh, out_delta_max_threshold, out_delta_min_threshold)
    dWhy = clip_values(dWhy, out_delta_max_threshold, out_delta_min_threshold)
    dbh = clip_values(dbh, out_delta_max_threshold, out_delta_min_threshold)
    dby = clip_values(dby, out_delta_max_threshold, out_delta_min_threshold)
     
    # Update the hidden state from this run.
    r.h = hs[length(ts)]

    return cost, dWxh, dWhh, dWhy, dbh, dby 

end

r = RNN(2,3,2)
r.Wxh = [.5 .2 ; 0.1 0.1 ; 0.2 0.2]
r.Whh = [.1 .1 .1 ; .2 .2 .2 ; 0.3 0.3 .3]
r.Why = [.4 .5 .6; .7 .8 .9 ]
r.h = [0.4 .2 0.8]'
println(forward_pass_backpropogate_batch(r, ([1 0]', [0 1]', [0 1]'), ([0 1]', [1 0]', [0 1]')))
println("\n",r.h)

# Hyperparamaters
hidden_size = 100           # In Karpathy's min-char-rnn.py, this is set to 100.
seq_length = 50             # In Karpathy's min-char-rnn.py, this is set to 25.
learning_rate = 1e-1        # In Karpathy's min-char-rnn.py, this is set to 1e-1.

"""
    update(r, xs, ts)

Train on a batch of inputs and truths, and update the model.

Performs Adagrad update gradient descent using the gradients calculated from backpropogation.

r - RNN
xs - a batch of inputs
ts - a batch of truths (as one-hot nx1 matrices)
"""
function update(r, xs, ts)
    loss, ∂cost_∂Wxh, ∂cost_∂Whh, ∂cost_∂Why, ∂cost_∂bh, ∂cost_∂by = forward_pass_backpropogate_batch(r,xs,ts)
    
    # Adagrad update (Gradient Descent)
    # As these terms grow, the shift below becomes smaller, because they're the denominator.
    r.mWxh += ∂cost_∂Wxh .^ 2
    r.mWhh += ∂cost_∂Whh .^ 2
    r.mWhy += ∂cost_∂Why .^ 2
    r.mbh += ∂cost_∂bh .^ 2
    r.mby += ∂cost_∂by .^ 2
    
    # Shift the weights down along their gradients (derivates).
    # Remember that ∂cost_∂Wxh specifies how much cost will *increase* with an
    # increase in Wxh, so to decrease cost, you would move along negative ∂cost_∂Wxh.
    r.Wxh -= learning_rate * ∂cost_∂Wxh  ./ sqrt(r.mWxh + 1e-8)
    r.Whh -= learning_rate * ∂cost_∂Whh  ./ sqrt(r.mWhh + 1e-8)
    r.Why -= learning_rate * ∂cost_∂Why  ./ sqrt(r.mWhy + 1e-8)
    r.bh -= learning_rate * ∂cost_∂bh    ./ sqrt(r.mbh + 1e-8)
    r.by -= learning_rate * ∂cost_∂by    ./ sqrt(r.mby + 1e-8)
    
    return loss
end

# Test Code:
r = RNN(2,3,2)
r.Wxh = [.5 .2 ; 0.1 0.1 ; 0.2 0.2]
r.Whh = [.1 .1 .1 ; .2 .2 .2 ; 0.3 0.3 .3]
r.Why = [.4 .5 .6; .7 .8 .9 ]
r.h = [0.4 .2 0.8]'
loss = update(r, ([1 0]', [0 1]', [0 1]', [1 0]'), ([0 1]', [1 0]', [0 1]', [1 0]'))
println(loss, r.h)

update(r, ([1 0]', [0 1]', [0 1]', [1 0]'), ([0 1]', [1 0]', [0 1]', [1 0]'))
update(r, ([1 0]', [0 1]', [0 1]', [1 0]'), ([0 1]', [1 0]', [0 1]', [1 0]'))
loss = update(r, ([1 0]', [0 1]', [0 1]', [1 0]'), ([0 1]', [1 0]', [0 1]', [1 0]'))

println(loss, r.h)

println(r.Wxh, r.Whh, r.Why, r.bh, r.by)

"""
    make_one_hot(length, index)

A helper function to make a "one-hot" vector, or a vector which is
all zeros with a one at the specified index. We will use this to
represent a character as input to the neural network.
"""
function make_one_hot(length, index)
    v = zeros(Float64, length, 1)
    v[index] = 1.0
    return v
end
x = make_one_hot(length(alphabet), reverse_alphabet['.'])
assert(1 == x[reverse_alphabet['.']])
assert(0 == x[reverse_alphabet['a']])

# Because Julia doesn't seem to have a function to sample from a set of probabilities?

"""
    rand_uniform(a, b)

Returns a number from the uniform random distribution between a and b.
"""
function rand_uniform(a, b)
    a + rand()*(b - a)
end
"""
    SampleFrom(probabilities)

Returns an index between 0 and length(probabilities), chosen based on the provided stepwise probabilities.
"""
function SampleFrom(probabilities)

    # Sum to create CDF:
    cdf = Array(Float64, 0)
    sum = 0.0
    for p in probabilities
        push!(cdf, sum + p)
        sum = cdf[end];
    end
        
    # Choose from CDF:
    cdf_value = rand_uniform(0.0,cdf[end])
    index = searchsortedfirst(cdf, cdf_value);

    return index;

end
# Test Code: run this repeatedly to see how the randomness follows the given distributions.
println(SampleFrom([1 1 1 1]))
println(SampleFrom([1 1 10 10]))
println(SampleFrom([0.1 0.2 0.7 0.8]))
#println(hist([SampleFrom([1 1 10 10]) for i in 0:1000]))

"""
    hallucinate(r, seed_idx, num_chars)

Generates, "hallucinates," text from the given RNN by repeatedly
1. running the network with an input,
2. generating an output probability distribution,
3. randomly selecting a new index from that distribution,
4. and using that index as a new input.
It does this num_chars times, to generate text num_chars long.
"""
function hallucinate(r, seed_idx, num_chars)
    hallucination = ""
    prev_ids = [seed_idx]
    # Should we clear the hidden state (not the weights)?
    # Note that Karpathy doesn't modify h, but I think it
    # makes sense to start over before each hallucination...
    # I don't do it here, though, to keep with his implementation.
    for x in range(1,num_chars)
        x_vec = make_one_hot(length(alphabet), prev_ids[end])
        y_norm,y,r.h = forward_pass_step(r, r.h, x_vec)

        # Now sample from y!
        letter_idx = SampleFrom(y_norm')
        #letter_idx = indmax(y)
        char = alphabet[letter_idx]
        
        append!(prev_ids,[letter_idx])
        hallucination = "$hallucination$char"
    end
    return hallucination
end
r = RNN(length(alphabet), 500, length(alphabet))
hallucinate(r, rand(1:length(alphabet)), 100)

training_rnn = RNN(length(alphabet), hidden_size, length(alphabet))
chars_trained = 1

i = 0
smooth_loss = -log(1.0/length(alphabet))*seq_length # loss at iteration 0
println("======== Iteration: $i, Chars Trained:", chars_trained-1, "/", length(original_text), " Cost: $smooth_loss ===========")
while true
    if chars_trained+seq_length+1 >= length(original_text)
        println("========= DONE!! ===========")
        break
    end
    x_vecs = [make_one_hot(length(alphabet), reverse_alphabet[x]) for x in original_text[chars_trained:chars_trained+seq_length]] 
    truth_vecs = [make_one_hot(length(alphabet), reverse_alphabet[y_]) for y_ in original_text[chars_trained+1:chars_trained+seq_length+1]]

    if i % 100 == 0
        println("======== Iteration: $i, Chars Trained:", chars_trained-1, "/", length(original_text), " Cost: $smooth_loss ===========")
        seed_idx = reverse_alphabet[original_text[chars_trained]]
        println(hallucinate(training_rnn, seed_idx, 100))
    end

    (loss,) = update(training_rnn, x_vecs, truth_vecs)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    chars_trained += seq_length
    i += 1

end

println(hallucinate(training_rnn, reverse_alphabet['\n'], 5000))  # Start with an endline just so it's pretty.



r = RNN(2,3,2)
r.Wxh = [.5 .2 ; 0.1 0.1 ; 0.2 0.2]
r.Whh = [.1 .1 .1 ; .2 .2 .2 ; 0.3 0.3 .3]
r.Why = [.4 .5 .6; .7 .8 .9 ]
r.h = [0.4 .2 0.8]'
outs = forward_pass_backpropogate_batch(r, ([1 0]', [0 1]'), ([0 1]', [1 0]'))
truth = (
[1.3988750128749587]',

[-0.02348709404610685 0.15845008784877856
 -0.029956754210863596 0.15314729364197688
 -0.02401725804279269 0.12098114381203691],

[0.08011354615577733 0.052773611291928826 0.06853660930090805
 0.07453013601361681 0.049556316201140885 0.06043836265216451
 0.058735290825127406 0.03907731268820138 0.04746229284518379],

[0.02188565806931217 -0.08201489754933375 -0.12198683957866582
 -0.021885658069312197 0.08201489754933372 0.1219868395786658],

[0.1349629938026717
 0.12319053943111329
 0.09696388576924422]'',

[-0.20382526255910166
 0.2038252625591016]'')

for i in range(1,length(outs))
    if outs[i] != truth[i]
        println("Incorrect! Outs:\n", outs[i], "\nvs\n", truth[i])
    end
end

true_r_h = [0.3344883118838543 0.37630407100318514 0.5673596773818216]'
if r.h != true_r_h
    println("Incorrect! r.h:\n", r.h, "\nvs\n", true_r_h)
end
