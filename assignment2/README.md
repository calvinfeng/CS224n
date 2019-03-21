# Assignment 2

## 1. Tensorflow Softmax

### 1a

`q1_softmax.py`

### 1b

`q1_softmax.py`

### 1c

Placeholder variables and feed dict work together to fill concrete values into the placeholder
variables.

### 1d

`q1_classifier.py`

#### 1e

Since TF computes gradients for us via its automatic differentiation, we don't need to define our
own gradients anymore.

## 2. Neural Transition-Based Dependency Parsing

### 2a

Let's remind ourselves what does each operation mean. Buffer is a FIFO queue, stack is a stack, and
dict is a dictionary that maps word to its dependencies, i.e. child word mapping to parent words.

- **SHIFT**: Remove first word from buffer queue and pushes it onto stack.
- **LEFT_ARC**: Add second item on the stack to the parents of first item of the stack, remove
  second item.
- **RIGHT_ARC**: Add first item on the stack to the parents of second item of the stack, remove
  first item.

|stack                           | buffer                                 | new dependency                   | transition |
|--------------------------------|----------------------------------------|----------------------------------|------------|
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                                  | SETUP      |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                                  | SHIFT      |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                                  | SHIFT      |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed: [I]                      | LEFT_ARC   |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                                  | SHIFT      |
| [ROOT, parsed, this, sentence] | [correctly]                            |                                  | SHIFT      |
| [ROOT, parsed, sentence]       | [correctly]                            | sentence: [this]                 | LEFT_ARC   |
| [ROOT, parsed]                 | [correctly]                            | parsed: [I, sentence]            | RIGHT_ARC  |
| [ROOT, parsed, correctly]      | []                                     |                                  | SHIFT      |
| [ROOT, parsed]                 | []                                     | parsed: [I, sentence, correctly] | RIGHT_ARC  |
| [ROOT]                         | []                                     | ROOT: [parsed]                   | RIGHT_ARC  |

### 2b

Notice that we need to shift N times for N words in a sentence. Every shift requires a `LEFT_ARC` or
a `RIGHT_ARC` to process the token. Therefore, we need 2N operations to complete the dependency
parsing of a sentence with N words.

### 2c

`q2_parser_transitions.py`

High level idea for each parse step.

```python
def shift(buffer, stack):
  val = buffer.pop(0)
  stack.append(val)


def left_arc(buffer, stack, parents):
  top = stack.pop()
  sec = stack.pop()
  if parents.get(top) is None:
    parents[top] = []
  parents[top].append(sec)
  stack.append(top) 


def right_arc(buffer, stack, parents):
  top = stack.pop()
  sec = stack.pop()
  if parents.get(sec) is None:
    parents[sec] = []
  parents[sec].append(top)
  stack.append(sec)
```

### 2d

`q2_parser_transitions.py`

The minibatch dependency parsing takes a list of sentences to be parsed, and a model that makes
parse decisions. Here's the pseudo code for the implementation.

```text
Initialize partial_parses as a list of PartialParse(s), one for each sentence in sentences.
Initialize unfinished_parses as a shallow copy of partial_parses.
while unfinished_parses is not empty do
    Take the first batch-size parses in unfinished_parses as a minibatch
    Use the model to predict the next transition for each partial parse in the minibatch
    Perform a parse step on each partial parse in the minibatch with its predicted transition
    Remove the completed parses from unfinished_parses
end while

Return dependencies for each parse in partial_parses
```

### 2e

`q2_initialization.py`

### 2f

Expected value is simply,

$$
P_{drop} * 0 + (1 - P_{drop}) * 1 * \gamma h_{i} = (1 - P_{drop})\gamma h_{i} = h_{i}
$$

Then,

$$
\gamma = \frac{1}{1 - P_{drop}}
$$

### 2g

The momentum prevents the gradient direction from bouncing around as it moves toward a local
optimum.

The parameters with the smallest graidents will get the larger updates. This will move the gradient
off a flat plateaus region.

### 2h

`q2_parser_model.py`

## 3. Recurrent Neural Netowkrs: Language Modeling

Since I've already done plenty LSTM/RNN gradient derivations from CS232n, I am not going to repeat
them here.