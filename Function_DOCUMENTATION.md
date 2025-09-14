# API Documentation - Custom GPT Model

This document provides detailed API documentation for the custom GPT model implementation, including class definitions, method signatures, and usage examples.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Training Pipeline](#training-pipeline)
3. [Data Processing](#data-processing)
4. [Deployment Interface](#deployment-interface)
5. [Configuration System](#configuration-system)
6. [Utilities](#utilities)

## Model Architecture

### `GPT` Class

The main model class implementing a decoder-only transformer architecture.

```python
class GPT(keras.Model):
    def __init__(self,
                 d_model: int = 128,
                 vocab_size: int = 94,
                 context_length: int = 512,
                 attention_heads: int = 8,
                 epsilon: float = 1e-5,
                 decoder_blocks: int = 3,
                 dropout_rate: float = 0.1,
                 **kwargs)
```

#### Parameters
- `d_model`: Embedding dimension (default: 128)
- `vocab_size`: Size of vocabulary (default: 94)
- `context_length`: Maximum sequence length (default: 512)
- `attention_heads`: Number of attention heads per layer (default: 8)
- `epsilon`: Layer normalization epsilon (default: 1e-5)
- `decoder_blocks`: Number of transformer blocks (default: 3)
- `dropout_rate`: Dropout probability (default: 0.1)

#### Methods

##### `call(inputs, training=False)`
Forward pass through the model.

**Parameters:**
- `inputs`: Tuple of `(token_ids, attention_mask)`
  - `token_ids`: `tf.Tensor` of shape `(batch_size, sequence_length)`
  - `attention_mask`: `tf.Tensor` of shape `(batch_size, sequence_length)`
- `training`: Boolean indicating training mode

**Returns:**
- `tf.Tensor`: Logits of shape `(batch_size, sequence_length, vocab_size)`

**Example:**
```python
# Initialize model
model = GPT(d_model=128, vocab_size=38, context_length=256)

# Prepare inputs
token_ids = tf.ones((1, 10), dtype=tf.int32)
attention_mask = tf.ones((1, 10), dtype=tf.int32)

# Forward pass
logits = model([token_ids, attention_mask])
print(f"Output shape: {logits.shape}")  # (1, 10, 38)
```

### `DecoderBlock` Class

Individual transformer block with self-attention and feed-forward layers.

```python
class DecoderBlock(keras.layers.Layer):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 epsilon: float = 1e-5,
                 **kwargs)
```

#### Parameters
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `dropout_rate`: Dropout probability
- `epsilon`: Layer normalization epsilon

### `SelfAttentionLayer` Class

Multi-head self-attention implementation with causal masking.

```python
class SelfAttentionLayer(keras.layers.Layer):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 **kwargs)
```

#### Methods

##### `call(inputs, attention_mask, training=False)`
Compute self-attention with causal masking.

**Parameters:**
- `inputs`: Input embeddings of shape `(batch_size, seq_len, d_model)`
- `attention_mask`: Attention mask of shape `(batch_size, seq_len)`
- `training`: Boolean for training mode

**Returns:**
- `tf.Tensor`: Attended representations of shape `(batch_size, seq_len, d_model)`

### `InitializePositionalEmbeddings` Class

Embedding layer with sinusoidal positional encoding.

```python
class InitializePositionalEmbeddings(keras.layers.Layer):
    def __init__(self,
                 d_model: int,
                 vocab_size: int,
                 CONTEXT_LEN: int = 128,
                 pad_value: int = 0,
                 **kwargs)
```

## Training Pipeline

### Training Script API

#### `train_model(config_path: str) -> None`

Main training function that orchestrates the entire training process.

**Parameters:**
- `config_path`: Path to configuration file

**Example:**
```python
from training.train import train_model

# Train model with configuration
train_model("configs/config_jane_austen_optimal.txt")
```

#### Training Configuration Format

Configuration files use a simple key-value format:

```
# Model Architecture
D_MODEL=128
CONTEXT_LEN=256
ATTENTION_HEADS=8
DECODER_BLOCKS=4
DROPOUT_RATE=0.1

# Training Parameters
LEARNING_RATE=0.0001
BATCH_SIZE=32
EPOCHS=100
VALIDATION_SPLIT=0.1

# Data Parameters
TOKENIZER_TYPE=char
TEXT_FILES=jane_austen_clean.txt
```

### Training Monitoring

#### `TrainingMonitor` Class

Handles real-time monitoring and logging during training.

```python
class TrainingMonitor:
    def __init__(self, experiment_dir: str)
    
    def log_epoch(self, epoch: int, metrics: dict) -> None
    def save_checkpoint(self, model: keras.Model, epoch: int) -> None
    def plot_training_curves(self) -> None
```

## Data Processing

### Text Preprocessing

#### `preprocess_text(text: str) -> str`

Clean and normalize input text.

**Parameters:**
- `text`: Raw text string

**Returns:**
- `str`: Cleaned text

**Example:**
```python
from data.data_processing import preprocess_text

raw_text = "Pride and Prejudice\n\nChapter 1\n\n"
clean_text = preprocess_text(raw_text)
```

### Tokenization

#### `CharacterTokenizer` Class

Character-level tokenizer for text processing.

```python
class CharacterTokenizer:
    def __init__(self, texts: List[str])
    
    def tokenize(self, text: str) -> List[int]
    def detokenize(self, tokens: List[int]) -> str
    def save(self, filepath: str) -> None
    def load(self, filepath: str) -> None
```

**Methods:**

##### `tokenize(text: str) -> List[int]`
Convert text to token IDs.

##### `detokenize(tokens: List[int]) -> str`
Convert token IDs back to text.

**Example:**
```python
from data.data_processing import CharacterTokenizer

# Initialize tokenizer
tokenizer = CharacterTokenizer(["Hello world", "How are you?"])

# Tokenize text
tokens = tokenizer.tokenize("Hello")
print(tokens)  # [8, 5, 12, 12, 15]

# Detokenize
text = tokenizer.detokenize(tokens)
print(text)  # "Hello"
```

### Dataset Creation

#### `create_tfrecord_dataset(text_files: List[str], output_dir: str, config: dict) -> None`

Create TFRecord files for efficient training.

**Parameters:**
- `text_files`: List of input text file paths
- `output_dir`: Directory to save TFRecord files
- `config`: Configuration dictionary

## Deployment Interface

### `GPTDeployer` Class

Handles model deployment and web interface.

```python
class GPTDeployer:
    def __init__(self, experiment_dir: str)
    
    def load_model_and_tokenizer(self) -> None
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7, top_k: int = 20) -> str
    def create_interface(self) -> gr.Blocks
```

#### Methods

##### `generate_text(prompt, max_length, temperature, top_k)`

Generate text using the trained model.

**Parameters:**
- `prompt`: Input text prompt
- `max_length`: Maximum characters to generate
- `temperature`: Sampling temperature (0.1-2.0)
- `top_k`: Top-k sampling parameter

**Returns:**
- `str`: Generated text

**Example:**
```python
from deployment.deploy import GPTDeployer

# Initialize deployer
deployer = GPTDeployer("experiments/jane_austen_proper_v1_20250914_093022")

# Generate text
result = deployer.generate_text(
    prompt="Elizabeth walked through the garden",
    max_length=100,
    temperature=0.8,
    top_k=25
)
print(result)
```

##### `create_improved_attention_visualization(text: str) -> PIL.Image`

Create attention heatmap visualization.

**Parameters:**
- `text`: Input text to analyze

**Returns:**
- `PIL.Image`: Attention heatmap image

### Deployment Functions

#### `launch_deployment(experiment_dir: str, share: bool = False, port: int = 7860) -> None`

Launch the Gradio web interface.

**Parameters:**
- `experiment_dir`: Path to trained model directory
- `share`: Whether to create public sharing link
- `port`: Port number for local server

**Example:**
```python
from deployment.deploy import launch_deployment

# Launch interface
launch_deployment(
    experiment_dir="experiments/jane_austen_proper_v1_20250914_093022",
    port=7861
)
```

## Configuration System

### Configuration Loading

#### `load_config(config_path: str) -> dict`

Load configuration from file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:**
- `dict`: Configuration dictionary

### Configuration Validation

#### `validate_config(config: dict) -> dict`

Validate and set defaults for configuration.

**Parameters:**
- `config`: Configuration dictionary

**Returns:**
- `dict`: Validated configuration

**Example:**
```python
from models.utils import load_config, validate_config

# Load and validate configuration
config = load_config("configs/config_jane_austen_optimal.txt")
config = validate_config(config)

print(f"Model dimension: {config['D_MODEL']}")
print(f"Context length: {config['CONTEXT_LEN']}")
```

## Utilities

### Text Processing Utilities

#### `prepare_sinusoidal_lookup_table(d_model: int, max_len: int) -> tf.Tensor`

Create sinusoidal positional encoding lookup table.

**Parameters:**
- `d_model`: Model dimension
- `max_len`: Maximum sequence length

**Returns:**
- `tf.Tensor`: Positional encoding table of shape `(max_len, d_model)`

#### `create_causal_mask(seq_len: int) -> tf.Tensor`

Create causal attention mask.

**Parameters:**
- `seq_len`: Sequence length

**Returns:**
- `tf.Tensor`: Causal mask of shape `(seq_len, seq_len)`

### Training Utilities

#### `CosineDecayWithWarmup` Class

Learning rate schedule with warmup and cosine decay.

```python
class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 warmup_steps: int,
                 total_steps: int,
                 peak_learning_rate: float = 1e-4,
                 min_learning_rate: float = 1e-6)
```

**Example:**
```python
from models.layers import CosineDecayWithWarmup

# Create learning rate schedule
lr_schedule = CosineDecayWithWarmup(
    warmup_steps=1000,
    total_steps=10000,
    peak_learning_rate=1e-4
)

# Use with optimizer
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
```

### Evaluation Utilities

#### `calculate_perplexity(model: keras.Model, dataset: tf.data.Dataset) -> float`

Calculate perplexity on validation dataset.

**Parameters:**
- `model`: Trained model
- `dataset`: Validation dataset

**Returns:**
- `float`: Perplexity score

#### `generate_samples(model: keras.Model, tokenizer: CharacterTokenizer, 
                     prompts: List[str]) -> List[str]`

Generate text samples for evaluation.

**Parameters:**
- `model`: Trained model
- `tokenizer`: Tokenizer instance
- `prompts`: List of input prompts

**Returns:**
- `List[str]`: Generated text samples

## Error Handling

### Common Exceptions

#### `ConfigurationError`
Raised when configuration is invalid or missing required parameters.

#### `ModelLoadError`
Raised when model loading fails due to incompatible weights or architecture.

#### `TokenizationError`
Raised when tokenization fails due to unsupported characters or encoding issues.

### Error Examples

```python
try:
    model = GPT.load_from_checkpoint("invalid/path")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")

try:
    config = load_config("missing_config.txt")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Performance Considerations

### Memory Usage
- Model size: ~3.2MB (801K parameters × 4 bytes)
- Training memory: ~2-4GB GPU memory for batch size 32
- Inference memory: ~500MB for single sequence generation

### Speed Optimization
- Use mixed precision training for faster training
- Batch inference for multiple generations
- TensorFlow's XLA compilation for production deployment

### Scaling Guidelines
- Increase `d_model` for more capacity (128 → 256 → 512)
- Add more `decoder_blocks` for deeper models (4 → 6 → 8)
- Adjust `context_length` based on use case (256 → 512 → 1024)

---

This API documentation provides comprehensive coverage of all public interfaces and methods in the custom GPT implementation. For implementation details and examples, refer to the source code in the respective modules.
