import tensorflow as tf
import keras
from typing import List, Dict, Tuple
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import prepare_sinusoidal_lookup_table

@keras.saving.register_keras_serializable()
class InitializePositionalEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        vocab_size : int,
        CONTEXT_LEN: int = 128,
        pad_value: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.pad_value = int(pad_value)
        self.vocab_size = vocab_size
        self._pos_table = prepare_sinusoidal_lookup_table(d_model, CONTEXT_LEN)

    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(
            name="embedding_matrix",
            shape=(self.vocab_size, self.d_model),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, text_batch):

        token_ids= text_batch # Unpacking Data Pre-processing inputs Embeddings
        
        # Embeddings lookup: (B, T, D)
        token_emb = tf.nn.embedding_lookup(self.embedding_matrix, token_ids)
        # Positional embeddings: slice and broadcast
        seq_len = tf.shape(token_ids)[1] # type: ignore
        pos_emb = self._pos_table[:seq_len, :]    # type: ignore # (T, D)
        pos_emb = tf.expand_dims(pos_emb, 0)     # (1, T, D)
        embeddings = token_emb + pos_emb         # (B, T, D)
        return embeddings

    # def compute_output_shape(self, input_shape):
    #     # input_shape: (batch_size,)
    #     batch = input_shape
    #     # Sequence length is dynamic: None
    #     return (batch, None, self.d_model), (batch, None)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            "pad_value": self.pad_value,
        })
        return cfg
    
    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, seq_len)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        return (batch_size, seq_len, self.d_model)

@keras.saving.register_keras_serializable()
class SelfAttentionLayer(keras.layers.Layer):
    def __init__(self, attention_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.attention_heads = attention_heads
        
    def build(self, input_shape):
        self.d_model = input_shape[0][-1]
        
        self.Query_projection = self.add_weight(
            name='Query_Vector_for_projection',
            initializer='random_normal',
            shape=(self.d_model, self.d_model),
            trainable=True 
        )
        self.Key_projection = self.add_weight(
            name='Key_Vector_for_projection',
            initializer='random_normal',
            shape=(self.d_model, self.d_model),
            trainable=True 
        )
        self.Value_projection = self.add_weight(
            name='Value_Vector_for_projection',
            initializer='random_normal',
            shape=(self.d_model, self.d_model),
            trainable=True 
        )
        self.output_projection = self.add_weight(
            name="Output_projection",
            initializer="random_normal",
            shape=(self.d_model, self.d_model),
            trainable=True,
        )

        self.d_head = self.d_model // self.attention_heads
        assert self.d_model % self.attention_heads == 0, "d_model must be divisible by attention_heads"

    def call(self, inputs):
        embeddings = inputs[0]
        token_masks = inputs[1]

        batch_size = tf.shape(embeddings)[0] # type: ignore
        seq_len = tf.shape(embeddings)[1] # type: ignore

        # 1. Project to Q, K, V
        Q = embeddings @ self.Query_projection
        K = embeddings @ self.Key_projection
        V = embeddings @ self.Value_projection

        # 2. Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, seq_len, self.attention_heads, self.d_head))
        K = tf.reshape(K, (batch_size, seq_len, self.attention_heads, self.d_head))
        V = tf.reshape(V, (batch_size, seq_len, self.attention_heads, self.d_head))

        Q = tf.transpose(Q, (0, 2, 1, 3))  # (batch, heads, seq_len, d_head)
        K = tf.transpose(K, (0, 2, 1, 3))
        V = tf.transpose(V, (0, 2, 1, 3))

        # 3. Compute attention scores
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, heads, seq_len, seq_len)
        scores = scores / tf.sqrt(tf.cast(self.d_head, tf.float32))
        
        # 4. FIXED MASKING - This was your main bug
        # 4a. Causal mask (L,L) lower triangular
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        
        # 4b. Token mask - FIXED: proper broadcasting to all heads
        token_mask = tf.cast(token_masks, tf.float32)  # (B, L)
        
        # Create proper attention mask shape (B, H, L, L)
        # Each head gets the same mask pattern
        attention_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, L, L)
        attention_mask = attention_mask * token_mask[:, tf.newaxis, tf.newaxis, :]  # type: ignore # (B, 1, 1, L)
        attention_mask = attention_mask * token_mask[:, tf.newaxis, :, tf.newaxis]  # type: ignore # (B, 1, L, 1)
        
        # Broadcast to all heads
        attention_mask = tf.broadcast_to(attention_mask, (batch_size, self.attention_heads, seq_len, seq_len))
        
        # 5. Apply mask with stronger negative value
        scores = tf.where(
            attention_mask > 0, 
            scores, 
            tf.constant(-1e30, dtype=scores.dtype)  # FIXED: Much more negative
        )

        # 6. Softmax and apply to values
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Add attention dropout (missing in your original)
        attention_weights = tf.nn.dropout(attention_weights, rate=0.1)
        
        context = attention_weights @ V   # (batch, heads, seq_len, d_head)
        
        # 7. Concatenate heads
        concat_context = tf.transpose(context, (0, 2, 1, 3))  # (batch, seq_len, heads, d_head)
        concat_context = tf.reshape(concat_context, (batch_size, seq_len, self.d_model))
        
        # 8. Final projection
        final_context = concat_context @ self.output_projection 
        return final_context
    
    def get_config(self):
        config = super().get_config()
        config.update({"attention_heads": self.attention_heads})
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.saving.register_keras_serializable()
class LayerNormalization(keras.layers.Layer):
    def __init__(self,eps=1e-5,**kwargs):
        super().__init__(**kwargs)
        self.eps = eps
    
    def build(self,input_shape): # Near Attention (batch, seq_len, d_model)
        self.alpha = self.add_weight(
            name = 'alpha',
            shape = input_shape[-1:],
            initializer = 'ones',
            dtype = tf.float32,
            trainable = True
        )
        self.beta = self.add_weight(
            name = 'beta',
            shape = input_shape[-1:],
            initializer = 'zeros',
            dtype = tf.float32,
            trainable = True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        normed = (inputs - mean) / tf.sqrt(var + self.eps) # type: ignore
        return self.alpha * normed + self.beta

    def get_config(self):
        base = super().get_config()
        return {**base, "eps": self.eps}
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class SimpleDense(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="he_normal",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, seq_len, input_dim)
        output = tf.matmul(inputs, self.kernel) + self.bias  # shape: (batch, seq_len, units))  # shape: (batch, seq_len, units)
        return output + self.bias

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, seq_len, input_dim)
        return (input_shape, input_shape[1], self.units)

    def get_config(self):
        base = super().get_config()
        return {**base, "units": self.units}
    
@keras.saving.register_keras_serializable()
class DecoderBlock(keras.Model):
    '''A single Decoder Block'''
    def __init__(self, d_model, n_heads, dropout_rate=0.1, epsilon=1e-5, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        # norms
        self.ln1 = LayerNormalization(epsilon)   # pre-attn
        self.ln2 = LayerNormalization(epsilon)   # pre-ffn
        # attention (assumes your SelfAttentionLayer accepts (x, attention_mask))
        self.attn = SelfAttentionLayer(n_heads)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        # FFN
        self.ffn1 = keras.layers.Dense(4 * d_model, activation="gelu")
        self.ffn2 = keras.layers.Dense(d_model)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, x, attention_mask, training=False):
        # Self-attention sublayer
        y = self.ln1(x)
        y = self.attn((y, attention_mask))          # shape: (B, T, d_model)
        y = self.dropout1(y, training=training)
        x = x + y                                    # residual

        # FFN sublayer
        y = self.ln2(x)
        y = self.ffn1(y)
        y = self.ffn2(y)
        y = self.dropout2(y, training=training)
        x = x + y                                    # residual
        return x
    
    def compute_output_shape(self, input_shape):
        # input_shape is typically (batch_size, seq_len, d_model)
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "epsilon": self.epsilon,
        })
        return config

@keras.saving.register_keras_serializable()
class GPT(keras.Model):
    '''
    GPT model with N distinct blocks
      -----------------------------------'''
    def __init__(self,
                 d_model: int = 128,
                 vocab_size: int = 94,
                 context_length: int = 512,
                 attention_heads: int = 8,
                 epsilon: float = 1e-5,
                 decoder_blocks: int = 3,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self._d_model = d_model
        self._vocab_size = vocab_size
        self._context_length = context_length
        self._attention_heads = attention_heads
        self._epsilon = epsilon
        self._decoder_blocks = decoder_blocks
        self._dropout_rate = dropout_rate

        # embeddings (yours)
        self.emb = InitializePositionalEmbeddings(
            d_model, vocab_size,context_length,name="init_embeddings"
        )

        # stack of distinct decoder blocks
        self.blocks = [
            DecoderBlock(d_model, attention_heads, dropout_rate, epsilon, name=f"decoder_block_{i}")
            for i in range(decoder_blocks)
        ]

        # final norm (GPT-2 style) and LM head
        self.final_ln = LayerNormalization(epsilon)
        self.lm_head = keras.layers.Dense(vocab_size, name="Model_head")

    def call(self, inputs, training=False):
        """
        inputs: (token_ids, attention_mask)
          - token_ids: int32 (B, T)
          - attention_mask: int32/float32 mask broadcasting to attention logits.
            Common shapes: (B, 1, 1, T) or (B, T) if your SelfAttentionLayer handles expansion.
        """
        token_ids, attention_mask = inputs
        x = self.emb(token_ids)                         # (B, T, d_model)

        for block in self.blocks:
            x = block(x, attention_mask, training=training)

        x = self.final_ln(x)
        logits = self.lm_head(x)                        # (B, T, vocab_size)
        return logits                                   # keep softmax outside

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self._d_model,
            "vocab_size": self._vocab_size,
            "context_length": self._context_length,
            "attention_heads": self._attention_heads,
            "epsilon": self._epsilon,
            "decoder_blocks": self._decoder_blocks,
            "dropout_rate": self._dropout_rate,
        })
        return cfg

@keras.saving.register_keras_serializable()
class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 warmup_steps: int,
                 total_steps: int,
                 peak_learning_rate: float = 1e-4,
                 min_learning_rate: float = 1e-6,
                 name: str = "cosine_decay_with_warmup"):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_learning_rate = peak_learning_rate
        self.min_learning_rate = min_learning_rate
        self.name = name
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase: linear increase from 0 to peak_learning_rate
        warmup_lr = self.peak_learning_rate * tf.math.divide(step, warmup_steps)
        
        # Cosine decay phase
        decay_steps = tf.math.subtract(total_steps, warmup_steps)
        progress = tf.math.divide(tf.math.subtract(step, warmup_steps), decay_steps)
        cosine_decay_lr = self.min_learning_rate + 0.5 * (
            self.peak_learning_rate - self.min_learning_rate
        ) * (1 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * progress))
        
        return tf.where(tf.less(step, warmup_steps), warmup_lr, cosine_decay_lr)
    
    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "peak_learning_rate": self.peak_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "name": self.name,
        }

# Example usage for your model
# Estimate your training parameters
EPOCHS = 20
STEPS_PER_EPOCH = 1000  # Adjust based on your dataset size and batch size
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = int(0.1 * TOTAL_STEPS)  # 10% warmup

# Create the learning rate schedule
lr_schedule = CosineDecayWithWarmup(
    warmup_steps=WARMUP_STEPS,
    total_steps=TOTAL_STEPS,
    peak_learning_rate=1e-4,  # Your desired peak learning rate
    min_learning_rate=1e-6    # Minimum learning rate at the end
)
