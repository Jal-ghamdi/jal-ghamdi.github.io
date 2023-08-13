# Understanding Recurrent Neural Networks (RNNs) in NLP: A High-Level Explanation
In the field of NLP, comprehending the nuances of language demands an understanding of context and sequence. Human language is not merely a compilation of words; rather, it forms a sequence in which the significance of each word is frequently shaped by the preceding words. This is precisely where RNNs come into play.

## The Essence of RNNs
Imagine reading a story where the unfolding plot is driven by what happened in previous chapters. The ability to connect events, sentiments, and meanings across time is what makes storytelling captivating. Similarly, RNNs simulate this temporal understanding in machines. They are the neural networks’ answer to memory and sequential comprehension, making them a cornerstone in the world of NLP.

## Inner Workings of RNNs
At its core, an RNN consists of two primary components: the input (the current word) and the hidden state (the encapsulated context). It processes the input and updates the hidden state at each step. Crucially, this updated hidden state then becomes the input for the next step, combined with the next word in the sequence. Through this iterative process, the RNN maintains a recollection of past information, enabling it to predict and generate coherent sequences. In other words, an RNN operates by processing sequential data in a step-by-step manner. At each time step, it takes an input, which could be a word or element in the sequence. Simultaneously, it considers the hidden state from the previous time step.
![image](https://github.com/Jal-ghamdi/jal-ghamdi.github.io/assets/44866137/3f4936e4-4883-4b56-9e06-ccfcc4e13519)

These two pieces of information are combined and passed through a neural network. The neural network’s parameters determine how this combination is transformed. The output of the neural network becomes the new hidden state, which is used in the next time step as the ‘previous’ hidden state.
![image](https://github.com/Jal-ghamdi/jal-ghamdi.github.io/assets/44866137/9eef1e1a-0f84-4be0-a805-b5e976ddd353)

This iterative process allows the RNN to capture dependencies and patterns in sequential data. In essence, an RNN ‘remembers’ previous steps and uses that information to influence its current prediction. Let’s have a look at the basic operations that occur within an RNN during each time step.

* Concatenate Inputs: concatenated_input $= [x_t, h_{t-1}]$

* Neural Network (NN) Calculation: $z_t = W_h \cdot$ concatenated_input $+ b$

* Apply the tanh activation function to $z_t​$ to get the hidden state $h_t$: $h_t = tanh(z_t)$

* Softmax and Output Calculation: output_probs = $softmax(W_o \cdot h_t + b)$

* Generating the Next Word: next_word = argmax(output_probs)

* Update Hidden State
  
* Repeat the Process

Continue to iterate through the steps for each subsequent time step, using the updated hidden state and the corresponding current word vector.

Let’s walk through a complete example of how an RNN generates the next word in a sequence. Let’s assume we have the following values for the current word vector $(x_t​)$ and the previous hidden state $(h_{t−1}​)$:

Assume the concatenated vector is $c = [0.2, 0.4, 0.6]$ for simplicity (i.e., concatenating two vectors involves simply sticking them together; for example, if $x = [1, 2, 3]$ and $h = [4, 5, 6]$ then the concatenated vector is $[1, 2, 3, 4, 5, 6]$). We’ll use the following weights and bias for the hidden state calculation:
<div>
  <p></p>
  <p>$$ w_h = \begin{bmatrix} 0.7 & -0.1 & 0.9 \\ -0.3 & 0.6 & -0.2 \\ 0.4 & 0.2 & 0.5 \end{bmatrix} $$</p>
</div>

<div>
  <p></p>
  <p>$$ b = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.3 \end{bmatrix} $$</p>
</div>

* First, we calculate $z_t$​ by multiplying the weights with the concatenated input and adding the bias:

<div>
  <p></p>
  <p>$$ z_t = w_h \cdot c + b $$</p>
</div>

* Performing the matrix multiplication and addition:

<div>
  <p></p>
  <p>$$ z_t = \begin{bmatrix} 0.72 \\ 0.06 \\ 0.54 \end{bmatrix} $$</p>
</div>


* Next, we apply the tanh activation function element-wise to the values of $z_t$​:
<div>
  <p></p>
  <p>$$ h_t = \tanh(z_t) = \begin{bmatrix} \tanh(0.72) \\ \tanh(0.06) \\ \tanh(0.54) \end{bmatrix} $$</p>
</div>

Let’s assume the calculated $h_t$​ is $[0.6200,0.0599,0.5010]$ (i.e., Remember the tanh function is a sigmoid-like function that maps input values to the range $[-1, 1]$. It produces output values between $-1$ and $1$].

Now, let's calculate the output. 
- Assuming we have a vocabulary of words, we’ll use another set of weights $W_o​$ and a bias $b_o​ for the output layer. Let’s say the vocabulary size is $4$.

<div>
  <p></p>
  <p>$$ w_o = \begin{bmatrix} 0.3 & -0.2 & 0.1 \\ 0.4 & 0.2 & 0.5 \\ -0.1 & 0.6 & 0.7 \\ 0.2 & 0.4 & -0.3 \end{bmatrix} $$</p>
</div>

<div>
  <p></p>
  <p>$$ b_o = \begin{bmatrix} 0.1 \\ 0.2 \\ -0.1 \\ 0.3 \end{bmatrix} $$</p>
</div>

- We’ll calculate the logits for each word in the vocabulary by multiplying $W_o$​ with ht​ and adding the bias $b_o$​:

<div>
  <p></p>
  <p>$$ logits = \begin{bmatrix} 0.3999 \\ 0.6068 \\ 0.4560 \\ 0.4639 \end{bmatrix} $$</p>
</div>

- We apply the softmax function to the logits to calculate the probabilities of each word:

<div>
  <p></p>
  <p>$$ \text{probabilities} = \text{softmax}(\text{logits}) = \begin{bmatrix} 0.2412 \\ 0.3199 \\ 0.2187 \\ 0.2202 \end{bmatrix} $$</p>
</div>

Choosing the word with the highest probability, which is the second word (index 1). Then, we update the hidden state ht​ as the new hidden state $h_{t−1}$​ for the next time step. So, in simple terms, the RNN takes an input $x_i$​, processes it through the hidden layer, and produces an output $y_i$. The hidden state $h_i$​ stores information about the input and the previous hidden state. This process is repeated for subsequent time steps, where the previous hidden state is used as input to compute the next hidden state and output. For a visual walkthrough of how information flows through an RNN, I recommend watching the video 

[https://www.youtube.com/embed/VIDEO_ID](https://youtu.be/CFN3yY6joh8)](https://youtu.be/CFN3yY6joh8)

## RNN’s Significance in NLP
The significance of RNNs in NLP cannot be overstated. Tasks like language translation, sentiment analysis, and text generation depend on grasping the sequential nature of language. However, RNNs have their limitations. For instance, they might struggle to maintain connections between distant words in a long text. This limitation paved the way for more advanced variations such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), which are engineered to capture these long-range dependencies.
In the upcoming blog post, I will talk about a more advanced and powerful variant of RNNs which is the LSTM network. While we’ve just explored the basics of how a simple RNN works, LSTM offers enhanced capabilities for handling longer sequences and capturing more complex dependencies within the data.



