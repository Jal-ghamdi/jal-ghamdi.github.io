# Long Short-Term Memory (LSTM)
LSTM stands as a sophisticated and transformative advancement within the realm of neural networks, particularly in the field of natural language processing (NLP). 
In essence, LSTM addresses the inherent challenge of modeling sequential data, such as language, by effectively capturing and retaining long-range dependencies within sequences. 
Unlike conventional recurrent neural networks (RNNs), which often struggle with vanishing or exploding gradients that hinder the learning of distant dependencies, LSTM introduces a dynamic 
memory cell and specialized gating mechanisms to circumvent these limitations. This allows LSTM to effectively capture context and relationships across time steps, making it particularly well-suited 
for tasks involving sequences of variable lengths. By integrating memory cells that retain information over extended intervals, and employing gating units that regulate the flow of information, LSTM 
has significantly enhanced the accuracy and efficiency of various natural language processing applications, marking it as a cornerstone in modern deep learning architectures.
Within the context of NLP, each word within a document's sequence is represented as a vector. As a result, LSTM operates on sequences of vectors corresponding to words in the document.
# Inner Workings of LSTM
RNNs Comprise three control neural networks - $f_n$, $i_n$, and $o_n$ - each network generates output vectors at time $n$ in the sequence. These networks follow a multi-layer perceptron structure, and the non-linear function for each control unit adopts a sigmoid function ($\sigma$), constraining the vector components to a range between zero and one. The Figure below shows the architecture of LSTM.
<img width="1172" alt="LSTM" src="https://github.com/Jal-ghamdi/jal-ghamdi.github.io/assets/44866137/eeecb0a2-1100-4f4c-bd54-2d0f4ca6972b">

First, the previous word $W_{n-1}$ and the previous hidden state $h_{n-1}$ are concatenated, resulting in the vector $X_{n-1}$. This vector is then fed separately into these neural networks. Apart from the hidden vector $h$, the LSTM model introduces a memory cell denoted as $c$. This memory cell is subject to modifications facilitated by a neural network (colored in orange) that employs a non-linear hyperbolic tangent function (tanh). This mechanism guarantees that the elements of $c$ fall within the spectrum of negative one to positive one. These four neural networks work in tandem to revise both the memory cell $c_n$ and the hidden variable $h_n$. This complex process encompasses numerous interactions, incorporating the utilization of $f_n$ and in to modify the memory cell $c_n$. Subsequently, the hyperbolic tangent is applied to transform the updated memory cell into the hidden unit $h_n$. This dynamic process showcases the sophisticated interplay of components within the LSTM architecture.
More specifically, The LSTM architecture consists of four distinct neural networks: $f_n$, $\tilde{C}_n$, $i_n$, and $o_n$. These networks process input data and contribute to the memory cell update and hidden state generation. Each of these neural networks uses different activation functions. For $f_n$, $i_n$, and $o_n$, the sigmoid function ($\sigma$) is applied. The sigmoid function squashes values to lie between 0 and 1, representing the extent of activation. For $\tilde{C}_n$, the hyperbolic tangent function (tanh) is employed, which maps values between -1 and 1.
** To update the memory cell ($c_n$), the process involves several steps:

- Forget Gate ($f_n$): The dot product of $f_n$ and the previous memory cell ($c_{n-1}$) results in values indicating which information to forget from the memory cell.

- Input Gate ($i_n$): The dot product of $\tilde{C}_n$ and i_n computes values that determine the new information to be added to the memory cell.

- Combining the Information: The outputs of the forget gate ($f_n$) and the input gate ($i_n$) are combined by element-wise multiplication and addition to obtain the updated memory cell ($c_n$).

** Hidden State Generation:
The generation of the hidden state ($h_n$) involves these steps:

- Output Gate ($o_n$): The dot product of $o_n$ and the input data generates values that determine the extent to which the memory cell's information contributes to the hidden state.

- Applying the Tanh Activation: The updated memory cell ($c_n$) is passed through the hyperbolic tangent function (tanh), which ensures that the values remain within the range of -1 to 1.

- Final Hidden State ($h_n$): A dot product between the tanh($c_n$) and $o_n$ yields the hidden state ($h_n$). This hidden state captures essential information from the input sequence and memory cell.

The LSTM architecture excels in capturing long-term dependencies within sequences, thanks to its incorporation of a memory cell and diverse control vectors. This enables the network to make informed decisions about retaining or discarding information across extended sequences. In essence, an LSTM network employs a combination of neural networks, activation functions, and gate mechanisms to modify its memory cell and produce hidden states. Through a sequence of operations involving forget, input, and output gates, combined with element-wise multiplications and additions, the network becomes capable of grasping and retaining patterns across different time spans. As a result, LSTM is highly effective for tasks that involve sequences and time series data. For a deeper understanding, refer to my YouTube video on the topic. <iframe width="560" height="315" src="https://www.youtube.com/embed/O_SwvSj-XkU?si=sWGxBzMweNaiMSje" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

