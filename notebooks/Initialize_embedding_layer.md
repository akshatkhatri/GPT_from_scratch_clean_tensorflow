We start with a batch of strings, for example:

tf.Tensor([b'Akshat Khatri', b'Hello World', b'Me'], shape=(3,), dtype=string)


This is just a list of 3 sentences, each represented as a TensorFlow string. The shape [3] tells us there are three items in the batch.

Next, each string is converted into token IDs using your lookup table. The function tokenize_and_build_token_id does this. It splits each string into characters, converts each character into a number, and pads the sequences so they all have the same length. That’s why your token IDs look like this:

[[27 66 74 63 56 75  1 37 63 56 75 73 64]
 [34 60 67 67 70  1 49 70 73 67 59  0  0]
 [39 60  0  0  0  0  0  0  0  0  0  0  0]]


The first row represents 'Akshat Khatri'.

The second row represents 'Hello World' and has two zeros at the end because it’s shorter than the first string.

The third row is 'Me' and is padded with zeros to match the length of the longest string.

To help the model know which positions are real tokens and which are padding, you create an attention mask:

[[1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 0 0]
 [1 1 0 0 0 0 0 0 0 0 0 0 0]]


1 means this position has a real token.

0 means this position is padding.

Transformers use this mask to ignore padding when computing attention.

Next, each token ID is converted into a vector using your embedding matrix:

token_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, tokens_in_id)


Your embedding matrix maps each token to a 128-dimensional vector.

After this, the shape is [batch_size, seq_len, embedding_dim], so [3, 13, 128] in this case.

Since transformers don’t know the order of tokens, you add positional encodings:

seq_len = tf.shape(tokens_in_id)[1]
pos_embeddings = self.pos_table[:seq_len, :]
pos_embeddings = tf.expand_dims(pos_embeddings, 0)


This ensures each token embedding carries information about its position in the sequence.

Broadcasting along the batch lets you add the same positional information to all items in the batch.

Finally, you combine token embeddings and positional embeddings:

embeddings = token_embeddings + pos_embeddings


Each token vector now contains both its identity and position information.

The final shape [3, 13, 128] is exactly what a transformer expects as input.

So in short, your process:

Start with a batch of strings.

Convert each character to a token ID.

Pad all sequences to the same length.

Create an attention mask to mark real tokens.

Look up each token ID in the embedding matrix.

Add positional embeddings.

Output is ready for a transformer: [batch_size, seq_len, embedding_dim].