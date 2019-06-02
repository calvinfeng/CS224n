# Assignment 3

## 1. A window into NER

### 1a

*Provide 2 examples of sentences containing a named entity with an ambiguous type.*

> Loki is going to Boulevard Pet Hospital on Monday for his skin treatment.
> Calvin needs to buy new suit from Zegna in the shopping mall.

*Why might it be important to use features apart from the word itself to predict named entity labels?*

The feature from word2vec only provides the meaning of a word, but it does not
consider the fact that some words can be used as named entity, like Backstreet
Boys. Backstreet literally means back of a street, the word itself does not
imply it is a named entity. We need to look at the whole context, like a window
of words to determine whether the word is actually a named entity.

*Describe at least two features that would help in predicting whether a word is part of a named entity*

* Context words surrounding a word
* Whether the word is capitalized

### 1b

*What are the dimensions of $e^{t}$, $W$, and $U$ if we use a window of size `w`?*

* Each word embedding vector is (1, D) and there are (2w + 1) of them, so it
  should be (1, (2w+1) * D))
* `W` has to be (2w + 1) * D, H)
* `U` has to be (H, C)

*What is the computational complexity of predicting labels for a sentence of length `T`?*

For a single window, it needs to get (2w + 1) vector look up from the embedding
weights. It then needs to perform computation, each element is at least touched
once. So it is O((2w + 1) * D * H). Finally it needs to perform softmax on each
class element, which is C. I suppose there are T windows. Then in total, I'd
guess O(T * (2w+1) * D * H + T*C).

### 1c

`q1_window.py`

### 1d

*Report your best development entity-level F1 score and the corresponding token-level confusion matrix.*

My F1 score is so fucked... I think I either downloaded the wrong data or my
model is not actually training.

```
go\gu   	PER     	ORG     	LOC     	MISC    	O       
PER     	1232.00 	12.00   	1461.00 	5.00    	439.00  
ORG     	796.00  	33.00   	763.00  	0.00    	500.00  
LOC     	968.00  	68.00   	822.00  	4.00    	232.00  
MISC    	586.00  	14.00   	546.00  	2.00    	120.00  
O       	35592.00	2379.00 	3589.00 	100.00  	1099.00 
```

*Describe at least 2 modeling limitations of the window-based model and support these conclusions with examples.*

One limitation is that it doesn't factor neighboring prediction into the model.
Suppose I have a sentence,

> When Captain Momoto decides to eat his cat food, he eats it like a boss.

If `Captain` was predicted to be a person, naturally `Momoto` should
automatically be part of a person or cat's name.

Second limitation is that it doesn't use attention model to look at other parts
of the sentence. If the model understands that `Momoto` is eating cat food,
`Momoto` must be a named entity. I think all these limitations will bring us to
the next topic, which is to use RNN for NER.

## 2. RNN for NER

### 2a

*How many more parameters does RNN model in comparison to the window-based model?*

It has one extra matrix which is the weights for hidden state vector. However, 
since we are not squashing multiple windows into one input, the size for W is
smaller in RNN case, it's `(D, H)` instead of `((2*w + 1) * D, H)`.

*What is the computational complexity of predicting labels for a sentence of length T*?

Linear time O(D) for computing embedded vectors, in fact I'd argue that it is a
constant time lookup for embedding vectors. O(H^2 + DH + H) for computing
each hidden state. Finally, it is O(HC + C) for computing the prediction. In
total it takes about O((D + H)HT), considering worst case, to compute labels for
the whole sentence.

### 2b

*Name at least one scenario in which decreasing the cross entropy cost would lead to an decrease in F1 scores*

Consider the case `The Backstreet Boys` again, the original prediction is

- Backstreet = MISC
- Boys = MISC

The trained model predcits

- Backstreet = MISC
- Boys = ORG

The entropy loss would decrease because it predicted one more label correctly
but the entity level F1 score would decrease because it predicted another named
entity incorrectly. That is, the precision goes down while recall remains the
same.

*Why is it difficult to directly optimize for F1?*

F1 is not differentiable, the model does not perform backpropgation to optimize
for it. Secondly, F1 requires predictions from the entire corpus to compute,
that makes it very difficult to batch and parallelize.

### 2c

`q2_rnn_cell.py`

### 2d

*How would loss and gradient updates change if we did not use masking?*

If we don't mask the loss, the extra 0-labels will be factored into the gradient
updates. The updates will affect the learning of the actual parameters. However,
if we zero out the loss due to the extra 0-labels, then we can prevent these 
redundant gradients from flowing back into our model.

### 2e

`q2_rnn.py`

### 2f

`q2_rnn.py`

### 2g

*Describe at least 2 modeling limitations of this RNN model.*

*For each limitation, suggest some way you could extend the model to overcome the limitation.*

One limitation is that the sequence is only making judgment based on historical
data, i.e. it looks at hidden state from the past but it doesn't look into the
future to influence its prediction. We can address this problem using a
bidirectional RNN.

Another limitation is that it doesn't adjust prediction of a particular token
based on adjacent predictions. We can use conditional random field loss penalize
the training if it doesn't let adjacent tokens to have the same tag.

## 3. Grooving With GRUs

### 3a Modeling Latching Behavior

Suppose we are given a sequence, with a starting digit 0 or 1, followed by N 0's.
We would like the hidden state to continue to remember what the first character
was, irrespective of how many 0s follow. In other words, once we see a 1, we want
its state to be 1 and stay 1.

Let's simplify our activation functions.

```python
def sigmoid(x):
    if x > 0:
        return 1
    return 0

def tanh(x):
    if x > 0:
        return 1
    return 0
```

#### RNN

Identify the values of `w_h, u_h, b_h` for an RNN cell that would allow it to
replicate the behavior above.

```python
h[t] = sigmoid(x[t].dot(u_h) + h[t-1].dot(w_h) + b_h)
```

There are four potential cases.

- `h[t-1] = 0` and `x[t] = 0`, we want `h[t] = 0`.
  - `sigmoid(b_h) = 0` therefore, `b_h <= 0`

- `h[t-1] = 0` and `x[t] = 1`, we want `h[t] = 1`.
  - `sigmoid(u_h + b_h) = 1` therefore, `u_h + b_h > 0`

- `h[t-1] = 1` and `x[t] = 0`, we want `h[t] = 1`.
  - `sigmoid(w_h + b_h) = 1` therefore, `w_h + b_h > 0`

- `h[t-1] = 1` and `x[t] = 1`, we want `h[t] = 1`.
  - `sigmoid(w_h + u_h + b_h) = 1` therefore, `w_h + u_h + b_h > 0`

We have four equations, and 3 unknowns. We can solve this easily.

#### GRU

Suppose that `w_r = u_r = b_r = b_z = b_h = 0` for GRU cell, then we can simplify
the equations to be the following.

```python
z[t] = sigmoid(x[t].dot(u_z) + h[t-1].dot(w_z))
r[t] = 0
h_hat[t] = tanh(x[t].dot(u_h))
h[t] = z[t] * h[t-1] + (1 - z[t]) * h_hat[t]
```

- `h[t-1] = 0` and `x[t] = 0`, we want `h[t] = 0`.
  - `z[t] = h_hat[t] = h[t] = 0`

- `h[t-1] = 0` and `x[t] = 1`, we want `h[t] = 1`.
  - `h[t] = (1 - sigmoid(u_z)) * tanh(u_h) = 1` therefore, `u_z <= 0` and `u_h > 0`

- `h[t-1] = 1` and `x[t] = 0`, we want `h[t] = 1`.
  - `h[t] = sigmoid(w_z) = 1` therefore, `w_z > 0` 

- `h[t-1] = 1` and `x[t] = 1`, we want `h[t] = 1`.
  - `h[t] = sigmoid(u_z + w_z) + (1 - sigmoid(u_z + w_z)) * tanh(u_h) = 1` therefore, 
    `u_z + w_z > 0` or `u_z + w_z <= 0` and `w_h` can be anything. 

Finally, we can conclude that `w_z > 0`, `w_h` can be anything, `u_z <= 0` and
`u_h > 0`.

### 3b Modeling Toggling Behavior

Suppose we are given a sequence, we want to produce a sequence that whenever it
sees 1, it switches all the following digits to 1 and when it sees 1 again, it
switches all the following digits to 0. This is like a toggle. Input `100100`
should produce `111000`. 

#### RNN

If the input is 0, the state will maintain its value, i.e. no toggling behavior.
The required condition would be the following.

- if `x[t] == 0`, `0 * w_h + 0 * u_h + b_h <= 0`
- if `x[t] == 1`, `1 * w_h + 0 * u_h + b_h > 0`

Therefore, `w_h` must be positive because `b_h` must be negative.

If the input is 1, the cell needs to flip state from 0 to 1 and when see 1
again, the cell needs to flip state from 1 back to 0. The required condition
would be the following.

- if `x[t] == 0` and `h[t] == 1`, `0 * w_h + 1 * u_h + b_h > 0`
- if `x[t] == 1` and `h[t] == 1`, `1 * w_h + 1 * u_h + b_h <= 0`

We can conclude that `w_h <= -u_h - b_h == w_h <= -(u_h + b_h) < 0`, which means
`w_h` must be negative. This is contradicting with the statement above. 

#### GRU

Let `w_r = u_r = b_z = b_h = 0`, what are the values of `b_r`, `w_z`, `w_h`, `u_z`
and `u_h` to perform the toggling behavior in GRU? 

We can do simple algebra again to confirm this. There are 5 unknowns and 4 
conditions. We have to manually pick one to be some random arbitrary number.
Let's say `b_r` is 1 to make our lives easier.

**Case 1**

`h[t-1] = x[t] = 1` and `h[t] = 0`

`sigmoid(u_z + w_z) + tanh(u_h + w_h)  <= sigmoid(u_z + w_z) * tanh(u_h + w_h)`

**Case 2**

`h[t-1] = h[t] = 1` and `x[t] = 0`

`sigmoid(w_z) + tanh(w_h) > sigmoid(w_z) * tanh(w_h)`

**Case 3**

`x[t] = h[t] = 1` and `h[t-1] = 0`

`sigmoid(u_z) + tanh(u_h) > sigmoid(u_z) * tanh(u_h)`

**Case 4**

`x[t] = h[t] = h[t-1] = 0`

Everything is zero. 0 <= 0

Now we can try random numbers to satisfy these conditions. For example,

- `u_z = = u_h = b_r = 1`
- `w_z = w_h = -1`

### 3c Implement GRU Cell

`q3_gru_cell.py`

### 3d Implement GRU

`q3_gru.py`