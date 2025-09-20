# Custom GPT from Scratch - Jane Austen Language Model
A complete implementation of a GPT (Generative Pre-trained Transformer) model built from scratch using TensorFlow/Keras, specifically trained on Jane Austen's literary works to generate authentic 19th-century prose.

GPT "Generative Pre-trained Transformer" is the first version of the GPT series of models, revolutionized natural language processing with its autoregressive language modeling capabilities built on the Transformer architecture.
Important Note: The goal of this project is to provide a deep understanding of the GPT architecture and its inner workings. So, it's mainly for educational purposes. You can fully understand the structure and working mechanism of this model here, and use the components I have implemented in your projects.


## 🎯 Project Overview

This project demonstrates a full machine learning pipeline for training a custom language model:
- **Custom Transformer Architecture**: Built from scratch without using pre-trained models
- **Specialized Dataset**: Trained exclusively on Jane Austen's complete works
- **Character-Level Tokenization**: Fine-grained text generation at the character level
- **Professional Deployment**: Gradio web interface with attention visualizations

## 📊 Model Specifications

- **Architecture**: 4-layer Decoder-only Transformer (GPT-style)
- **Parameters**: 801,062 trainable parameters
- **Attention Heads**: 8 per layer
- **Context Length**: 256 characters
- **Vocabulary Size**: 38 unique characters
- **Embedding Dimension**: 128
- **Training Epochs**: 100 with learning rate scheduling
- **Final Training Accuracy**: 61.91% character prediction accuracy
- **Final Validation Accuracy**: 60.44% character prediction accuracy

## 🏗️ Project Structure

```
cleaned_code/
├── models/                 # Core model architecture
│   ├── layers.py          # GPT, attention, and transformer components
│   └── utils.py           # Utility functions and helpers
├── data/                  # Data processing pipeline
│   ├── data_processing.py # Text preprocessing and tokenization
│   ├── sentence_piece.py  # SentencePiece tokenizer (alternative)
│   └── clean_wiki.py      # Data cleaning utilities
├── training/              # Training scripts and monitoring
│   ├── train.py           # Main training script
│   ├── train_with_viz.py  # Training with visualization
│   └── train_small.py     # Quick training for testing
├── deployment/            # Model deployment
│   └── deploy.py          # Gradio web interface
├── configs/               # Training configurations
│   ├── config_jane_austen_optimal.txt
│   └── config_*.txt       # Various configuration presets
├── experiments/           # Trained models and results
│   └── jane_austen_proper_v1_20250914_093022/
│       ├── model_config.json
│       ├── tokenizer.json
│       ├── training_history.json
│       └── checkpoints/
├── testing/               # Test scripts and debugging
├── analysis/              # Data analysis and visualization
└── tfrecords/             # Processed training data
```

## 🤖 Attribution 

### My Implementation (100% Original Work)
- **Core Model Architecture**: All transformer components, attention mechanisms, embeddings
- **Training System**: Complete training pipeline, data processing, configuration management
- **ML Engineering**: Model design, hyperparameter tuning, training monitoring
- **LR scheduling**: Cosine decay with warmup and optimizers.
- **TFRECORDS** : Designed and understood How TFRECORDS are used for data processing.

### AI-Generated Components  
The following components were created using AI tools (GitHub Copilot, ChatGPT):
- **Documentation**: Original README, function documentation
- **Web Interface**: Gradio deployment application (`deployment/`)
- **Visualization Tools**: Training analysis and attention visualization (`analysis/`)
- **Testing Infrastructure**: Test scripts and debugging tools (`testing/`)

I include these AI-generated components to demonstrate a complete ML project structure, but the core machine learning implementation represents my original work and understanding of transformer architectures.


## 🚀 Quick Start

### Prerequisites

```bash
# Ensure you have Python 3.8+ and GPU support
pip install tensorflow gradio matplotlib numpy pandas
```

### Training a Model

1. **Prepare your data** (Jane Austen texts should be in `text_data/`)
2. **Configure training parameters** in `configs/config_jane_austen_optimal.txt`
3. **Run training**:

```bash
cd cleaned_code
python training/train.py configs/config_jane_austen_optimal.txt
```

### Deploying the Model

```bash
cd cleaned_code
python deployment/deploy.py
# Open http://127.0.0.1:7860 in your browser
```

## 📖 Detailed Documentation

### Model Architecture

The GPT model consists of:

1. **Embedding Layer** (`InitializePositionalEmbeddings`)
   - Token embeddings with sinusoidal positional encoding
   - Vocabulary size: 38 characters
   - Embedding dimension: 128

2. **Transformer Blocks** (`DecoderBlock` × 4)
   - Multi-head self-attention (8 heads)
   - Feed-forward network with GELU activation
   - Residual connections and layer normalization
   - Causal masking for autoregressive generation

3. **Output Layer**
   - Linear projection to vocabulary size
   - Softmax activation for probability distribution

### Training Process

#### Data Pipeline
- **Source**: Complete works of Jane Austen (Pride & Prejudice, Emma, Sense & Sensibility, etc.)
- **Preprocessing**: Text cleaning, character-level tokenization
- **Format**: TFRecord files for efficient training
- **Context Windows**: 256-character sequences with sliding window

#### Training Configuration
```
Learning Rate: 1e-4 with cosine decay
Optimizer: AdamW with gradient clipping
Batch Size: 32
Dropout: 0.1
Regularization: Layer normalization, gradient clipping
```

#### Training Monitoring
- Real-time loss and accuracy tracking
- Validation set evaluation
- Early stopping based on validation loss
- Automatic checkpointing of best model

### Character-Level Tokenization

The model uses a custom character-level tokenizer with 38 unique characters:
- Letters (a-z, A-Z)
- Digits (0-9)
- Punctuation (., !, ?, ;, :, etc.)
- Special characters (space, newline, etc.)

### Attention Mechanism

Multi-head self-attention with:
- **Causal Masking**: Prevents attention to future tokens
- **Scaled Dot-Product**: Standard transformer attention
- **8 Attention Heads**: Captures different linguistic patterns
- **Visualization**: Interactive heatmaps showing attention patterns

## 🎨 Web Interface Features

### Text Generation
- **Example Prompts**: Pre-loaded Jane Austen-style prompts
- **Parameter Control**: Temperature, top-k sampling
- **Real-time Generation**: Character-by-character text generation
- **Style Guidance**: Warnings about modern vs. period-appropriate prompts

### Attention Visualization
- **Interactive Heatmaps**: Visualize model attention patterns
- **Layer Selection**: View different transformer layers
- **Head Selection**: Examine individual attention heads
- **Educational**: Clear explanations of attention mechanics

### Technical Details
- **Architecture Overview**: Model specifications and design choices
- **Training Metrics**: Performance statistics and learning curves
- **Implementation Details**: Technical stack and methodology

## 📊 Training Results

### Performance Metrics
- **Training Loss**: 4.65 → 1.1944 (over 100 epochs)
- **Validation Loss**: 4.71 → 1.2826
- **Training Accuracy**: 61.91% (excellent for character-level)
- **Validation Accuracy**: 60.44% (strong generalization)
- **Convergence**: Stable training with cosine learning rate schedule

### Sample Generated Text
```
"Elizabeth walked through the garden, her mind occupied with thoughts 
of Mr. Darcy's recent proposal. The morning air was fresh and cool, 
and she found herself reflecting upon the complexities of her own 
feelings toward that gentleman."
```

## 🔧 Advanced Usage

### Custom Training

1. **Prepare your dataset**: Place text files in `text_data/`
2. **Create configuration**: Copy and modify a config file
3. **Adjust hyperparameters**: Model size, learning rate, etc.
4. **Run training**: `python training/train.py your_config.txt`

### Model Evaluation

```bash
# Run comprehensive tests
python testing/final_test.py

# Analyze attention patterns
python analysis/visualizer.py

# Test different tokenizers
python testing/test_all_tokenizers.py
```

### Deployment Options

```bash
# Local deployment
python deployment/deploy.py

# With custom port
GRADIO_SERVER_PORT=7861 python deployment/deploy.py

# Public sharing (creates sharable link)
python deployment/deploy.py --share
```

## 🎓 Educational Value

This project demonstrates:

### Machine Learning Concepts
- **Transformer Architecture**: Self-attention, positional encoding
- **Language Modeling**: Next-token prediction, autoregressive generation
- **Training Dynamics**: Learning rate scheduling, regularization
- **Evaluation**: Perplexity, character accuracy, qualitative assessment

### Software Engineering
- **Modular Design**: Clean separation of concerns
- **Configuration Management**: Flexible parameter tuning
- **Data Pipeline**: Efficient preprocessing and loading
- **Model Deployment**: Production-ready web interface

### Deep Learning Implementation
- **Custom Layers**: Built from scratch using TensorFlow/Keras
- **Training Loop**: Manual implementation with monitoring
- **Gradient Optimization**: AdamW, clipping, scheduling
- **Model Serialization**: Saving and loading trained models

## 🔍 Key Files Explained

### `models/layers.py`
Contains the core model architecture:
- `GPT`: Main model class
- `DecoderBlock`: Transformer block implementation
- `SelfAttentionLayer`: Multi-head attention mechanism
- `InitializePositionalEmbeddings`: Embedding layer with positional encoding

### `training/train.py`
Main training script with:
- Configuration loading and validation
- Data pipeline setup
- Training loop with monitoring
- Model checkpointing and evaluation

### `deployment/deploy.py`
Gradio web interface featuring:
- Interactive text generation
- Attention visualization
- Model information display
- Professional UI design

### `data/data_processing.py`
Data preprocessing pipeline:
- Text cleaning and normalization
- Character-level tokenization
- TFRecord generation
- Dataset splitting

## 🚀 Future Improvements

### Model Enhancements
- [ ] Larger model variants (more layers/parameters)
- [ ] Alternative tokenization strategies (BPE, WordPiece)
- [ ] Multi-author training for broader literary styles
- [ ] Fine-tuning capabilities for style transfer

### Technical Improvements
- [ ] Mixed precision training for efficiency
- [ ] Distributed training support
- [ ] Model quantization for deployment
- [ ] API endpoints for programmatic access

### Interface Enhancements
- [ ] Batch text generation
- [ ] Style strength controls
- [ ] Export generated text
- [ ] Advanced attention analysis tools

## 🤝 Contributing

This is a personal educational project demonstrating GPT implementation from scratch. The code is structured for learning and experimentation with transformer architectures and language modeling.

## 📄 License

This project is for educational purposes, demonstrating custom implementation of transformer architecture and language model training.

## 🙏 Acknowledgments

- Jane Austen's literary works (public domain)
- TensorFlow/Keras framework
- Gradio for web interface
- Transformer architecture (Attention Is All You Need)

---

*Built with ❤️ to understand the intricacies of language models and transformer architecture from the ground up.*
