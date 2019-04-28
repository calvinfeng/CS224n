# Assignment 3

## 1. A window into NER

### 1a

*Provide 2 examples of sentences containing a named entity with an ambiguous type.*

> Loki is going to Boulevard Pet Hospital on Monday for his skin treatment.
> Calvin needs to buy new suit from Zegna in the shopping mall.

*Why might it be important to use features apart from the word itself to predict named entity labels?*

The feature from word2vec only provides the meaning of a word, but it does not consider the
fact that some words can be used as named entity, like Backstreet Boys. Backstreet literally
means back of a street, the word itself does not imply it is a named entity. We need to look at
the whole context, like a window of words to determine whether the word is actually a named
entity.

*Describe at least two features that would help in predicting whether a word is part of a named entity*

* Context words surrounding a word
* Whether the word is capitalized

### 1b

*What are the dimensions of $e^{t}$, $W$, and $U$ if we use a window of size `w`?*

* Each word embedding vector is (1, D) and there are (2w + 1) of them, so it should be (1, (2w+1) * D))
* `W` has to be (2w + 1) * D, H)
* `U` has to be (H, C)

*What is the computational complexity of predicting labels for a sentence of length `T`?*

For a single window, it needs to get (2w + 1) vector look up from the embedding weights. It then
needs to perform computation, each element is at least touched once. So it is O((2w + 1) * D * H).
Finally it needs to perform softmax on each class element, which is C. I suppose there are T windows.
Then in total, I'd guess O(T * (2w+1) * D * H + T*C).

### 1c

`q1_window.py`

### 1d

*Report your best development entity-level F1 score and the corresponding token-level confusion matrix.*

My F1 score is so fucked... I think I either downloaded the wrong data or my model is not actually
training.

```
go\gu   	PER     	ORG     	LOC     	MISC    	O       
PER     	1232.00 	12.00   	1461.00 	5.00    	439.00  
ORG     	796.00  	33.00   	763.00  	0.00    	500.00  
LOC     	968.00  	68.00   	822.00  	4.00    	232.00  
MISC    	586.00  	14.00   	546.00  	2.00    	120.00  
O       	35592.00	2379.00 	3589.00 	100.00  	1099.00 
```

*Describe at least 2 modeling limitations of the window-based model and support these conclusions with examples.*

One limitation is that it doesn't factor neighboring prediction into the model. Suppose I have a
sentence,

> When Captain Momoto decides to eat his cat food, he eats it like a boss.

If `Captain` was predicted to be a person, naturally `Momoto` should automatically be part of a
person or cat's name.

Second limitation is that it doesn't use attention model to look at other parts of the sentence.
If the model understands that `Momoto` is eating cat food, `Momoto` must be a named entity. I think
all these limitations will bring us to the next topic, which is to use RNN for NER.