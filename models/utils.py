import tensorflow as tf
import keras
import numpy as np
import os
import re
from typing import List, Dict, Tuple, Union, Any

@tf.function
@keras.saving.register_keras_serializable()
def prepare_sinusoidal_lookup_table(EMBEDDING_SIZE: int = 128, max_seq_len: int = 512):
    """
    Builds a sinusoidal positional encoding lookup table.
    
    Args:
      EMBEDDING_SIZE: dimensionality of each position encoding vector (must be even).
      max_seq_len: maximum sequence length (number of positions).
    
    Returns:
      lookup_table: a tf array of shape (max_seq_len, EMBEDDING_SIZE)
                    where row p gives the positional encoding for position p.
    """
    # Initialize the table
    lookup_table = np.zeros((max_seq_len, EMBEDDING_SIZE), dtype=np.float32)
    
    # Compute the angle rates for each dimension
    # angle_rates[k] = 1 / (10000^(2*(k//2) / EMBEDDING_SIZE))
    dims = np.arange(EMBEDDING_SIZE)[np.newaxis, :]   # shape (1, EMBEDDING_SIZE)
    positions = np.arange(max_seq_len)[:, np.newaxis] # shape (max_seq_len, 1)
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / EMBEDDING_SIZE)
    
    # Compute the angle for each position and dimension: position * angle_rate
    angle_rads = positions * angle_rates  # shape (max_seq_len, EMBEDDING_SIZE)
    
    # Apply sin to even indices (0,2,4,...) and cos to odd indices (1,3,5,...)
    lookup_table[:, 0::2] = np.sin(angle_rads[:, 0::2])
    lookup_table[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return tf.constant(lookup_table)

@keras.saving.register_keras_serializable()
def tokenize_and_build_vocabulary_tf(file_path_list: List[str], existing_vocab: Dict[str, int] | None = None) -> Dict[str, int]:
    """
    Build a character-level vocabulary dictionary from text files.
    
    Args:
        file_path_list: List of file paths containing the text corpus.
        existing_vocab: Optional existing vocabulary to extend.

    Returns:
        token_to_id: dict mapping character to unique integer token ID.
    """
    if isinstance(file_path_list, (str, bytes)):
        file_path_list = [file_path_list] # type: ignore
    if existing_vocab is None:
        existing_vocab = {}
    vocab_set = set(existing_vocab.keys())
    
    for file_name in file_path_list:
        if os.path.isdir(file_name):
            raise IsADirectoryError(f"Expected file path, got directory: {file_name}")
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        with open(file_name, encoding="utf-8") as f:
            text = f.read()
            vocab_set.update(text)
    
    sorted_tokens = sorted(vocab_set)
    token_to_id = {char: idx for idx, char in enumerate(sorted_tokens)}
    return token_to_id

@keras.saving.register_keras_serializable()
def build_id_to_token(token_to_id: dict[str, int]) -> dict[int, str]:
    """
    Convert a token_to_id dictionary into an id_to_token dictionary.

    Args:
        token_to_id: Dictionary mapping characters (tokens) to unique IDs.

    Returns:
        id_to_token: Dictionary mapping IDs back to characters (tokens).
    """
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return id_to_token


@keras.saving.register_keras_serializable()
def tokenize_and_build_token_id(token_to_id_dict: Dict[str, int], text_batch: List[str], max_seq_len: int, pad_value: int = 0) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Tokenize a batch of text strings into character token IDs using a token dictionary,
    then pad/truncate to max_seq_len and create attention masks.

    Args:
        token_to_id_dict: dict mapping character to integer token ID.
        text_batch: list of text strings to tokenize.
        max_seq_len: maximum sequence length after padding/truncation.
        pad_value: integer ID used for padding tokens.

    Returns:
        token_ids: tf.Tensor of shape (batch_size, max_seq_len), dtype tf.int32.
        attention_mask: tf.Tensor of shape (batch_size, max_seq_len), dtype tf.int32 (1 for real tokens, 0 for padding).
    """
    batch_token_ids = []
    for text in text_batch:
        ids = [token_to_id_dict.get(c, pad_value) for c in text]
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        else:
            ids += [pad_value] * (max_seq_len - len(ids))
        batch_token_ids.append(ids)
    
    token_ids = np.array(batch_token_ids, dtype=np.int32)
    attention_mask = (token_ids != pad_value).astype(np.int32)
    
    return tf.constant(token_ids), tf.constant(attention_mask) # type: ignore

def format_model_summary(model):
    print("="*80)
    print(f"{'GPT MODEL SUMMARY':^80}")
    print("="*80)
    
    total_params = model.count_params()
    total_layers = len(model.layers)
    total_weights = sum(len(layer.trainable_weights) for layer in model.layers)
    try:
        output_shapes = [tuple(out.shape) for out in model.outputs]
    except Exception:
        output_shapes = ["(unavailable)"]
    
    print(f"{'Total parameters:':<22} {total_params:,}")
    print(f"{'Total layers:':<22} {total_layers}")
    print(f"{'Trainable weights:':<22} {total_weights}")
    print(f"{'Final output shape(s):':<22} {output_shapes if output_shapes else '(N/A)'}")
    print("-"*80)
    
    header = f"{'Idx':>3} | {'Layer Type':<24} | {'Layer Name':<23} | {'Weight Name':<28} | {'Shape':<15} | {'Params':>8}"
    print(header)
    print("-" * len(header))
    
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        layer_name = layer.name
        weights = layer.trainable_weights
        layer_weight_count = len(weights)

        if layer_weight_count == 0:
            print(f"{i:03} | {layer_type:<24} | {layer_name:<23} | {'-':<28} | {'-':<15} | {'0':>8}")
        else:
            for j, w in enumerate(weights):
                n = int(np.prod(w.shape)) if hasattr(w, "shape") else "?"
                shape_str = str(tuple(w.shape))
                weight_name = w.name
                if j == 0:
                    print(f"{i:03} | {layer_type:<24} | {layer_name:<23} | {weight_name:<28} | {shape_str:<15} | {n:>8,}")
                else:
                    print(f"    | {'':<24} | {'':<23} | {weight_name:<28} | {shape_str:<15} | {n:>8,}")
        if layer_weight_count > 1:
            print(f"    | {'':<24} | {'':<23} | {'':<28} | {'':<15} | {'':>8}")

    print("="*80)
    print("Note: Only trainable weights are listed above. Output shapes may be unavailable\n"
          "for subclassed models or models not built symbolically.")
    print("="*80 + "\n")

def load_config(config_path):
    """Load configuration from text file"""
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Split on first '=' and handle inline comments
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove inline comments (everything after #)
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    
                    # Convert to appropriate type
                    if value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                        if '.' in value or 'e' in value.lower():
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    else:
                        config[key] = value
    return config


# ==================== WORD-LEVEL TOKENIZATION ====================

@keras.saving.register_keras_serializable()
def tokenize_and_build_vocabulary_word(file_path_list: Union[List[str], str], vocab_size: int = 10000, 
                                     existing_vocab: Dict[str, int] | None = None) -> Dict[str, int]:
    """
    Build a word-level vocabulary dictionary from text files.
    
    Args:
        file_path_list: List of file paths containing the text corpus.
        vocab_size: Maximum vocabulary size (most frequent words).
        existing_vocab: Optional existing vocabulary to extend.

    Returns:
        token_to_id: dict mapping word to unique integer token ID.
    """
    if isinstance(file_path_list, (str, bytes)):
        file_path_list = [file_path_list]  # type: ignore
    if existing_vocab is None:
        existing_vocab = {}
    
    word_counts = {}
    
    for file_name in file_path_list:
        if os.path.isdir(file_name):
            raise IsADirectoryError(f"Expected file path, got directory: {file_name}")
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        
        with open(file_name, encoding="utf-8") as f:
            text = f.read()
            # Simple word tokenization (split on whitespace and punctuation)
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Add special tokens
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    token_to_id = {}
    
    # Add special tokens first
    for i, token in enumerate(special_tokens):
        token_to_id[token] = i
    
    # Sort words by frequency and take top vocab_size - special_tokens
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    max_words = vocab_size - len(special_tokens)
    
    for i, (word, count) in enumerate(sorted_words[:max_words]):
        token_to_id[word] = len(special_tokens) + i
    
    return token_to_id


@keras.saving.register_keras_serializable()
def tokenize_text_word(text: str, token_to_id: Dict[str, int]) -> List[int]:
    """
    Tokenize text using word-level tokenization.
    
    Args:
        text: Input text string.
        token_to_id: Word-to-ID mapping dictionary.
    
    Returns:
        List of token IDs.
    """
    import re
    words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    unk_id = token_to_id.get('<UNK>', 1)
    return [token_to_id.get(word, unk_id) for word in words]


# ==================== SENTENCEPIECE TOKENIZATION ====================

def train_and_load_sentencepiece(file_path_list: Union[List[str], str], vocab_size: int = 2000, 
                                model_prefix: str = 'spm_gpt') -> Any:
    """
    Train and load SentencePiece tokenizer from text files.
    
    Args:
        file_path_list: List of file paths containing the text corpus.
        vocab_size: Vocabulary size for SentencePiece.
        model_prefix: Prefix for model files.
    
    Returns:
        SentencePiece processor.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("sentencepiece is required for subword tokenization. Install with: pip install sentencepiece")
    
    if isinstance(file_path_list, (str, bytes)):
        file_path_list = [file_path_list]  # type: ignore
    
    # Create input file list for SentencePiece training
    input_files = ','.join(file_path_list)
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',  # Byte Pair Encoding
        normalization_rule_name='identity',
        remove_extra_whitespaces=False,
        input_sentence_size=2000000,
        shuffle_input_sentence=True
    )
    
    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    
    return sp


@keras.saving.register_keras_serializable()
def tokenize_text_sentencepiece(text: str, sp_processor) -> List[int]:
    """
    Tokenize text using SentencePiece.
    
    Args:
        text: Input text string.
        sp_processor: SentencePiece processor.
    
    Returns:
        List of token IDs.
    """
    return sp_processor.encode_as_ids(text)


def sentencepiece_to_token_dict(sp_processor) -> Dict[str, int]:
    """
    Create token-to-ID dictionary from SentencePiece processor.
    
    Args:
        sp_processor: SentencePiece processor.
    
    Returns:
        Dictionary mapping tokens to IDs.
    """
    vocab_size = sp_processor.get_piece_size()
    token_to_id = {}
    
    for i in range(vocab_size):
        token = sp_processor.id_to_piece(i)
        token_to_id[token] = i
    
    return token_to_id


# ==================== UNIFIED TOKENIZER FACTORY ====================

def create_tokenizer(config: Dict, file_path_list: List[str]):
    """
    Create appropriate tokenizer based on configuration.
    
    Args:
        config: Configuration dictionary containing TOKENIZER_TYPE and VOCAB_SIZE.
        file_path_list: List of file paths for training tokenizer.
    
    Returns:
        Tuple of (token_to_id_dict, tokenize_function, additional_info)
    """
    tokenizer_type = config.get('TOKENIZER_TYPE', 'char').lower()
    vocab_size = config.get('VOCAB_SIZE', 2000)
    
    if tokenizer_type == 'char':
        print("Using character-level tokenization...")
        token_to_id = tokenize_and_build_vocabulary_tf(file_path_list)
        
        def tokenize_func(text):
            pad_id = token_to_id.get('\0', 0)  # Use null char as pad, or 0
            return [token_to_id.get(char, pad_id) for char in text]
        
        return token_to_id, tokenize_func, None
    
    elif tokenizer_type == 'word':
        print(f"Using word-level tokenization with vocab_size={vocab_size}...")
        token_to_id = tokenize_and_build_vocabulary_word(file_path_list, vocab_size)
        
        def tokenize_func(text):
            return tokenize_text_word(text, token_to_id)
        
        return token_to_id, tokenize_func, None
    
    elif tokenizer_type == 'sentencepiece':
        print(f"Using SentencePiece tokenization with vocab_size={vocab_size}...")
        sp_processor = train_and_load_sentencepiece(file_path_list, vocab_size)
        token_to_id = sentencepiece_to_token_dict(sp_processor)
        
        def tokenize_func(text):
            return tokenize_text_sentencepiece(text, sp_processor)
        
        return token_to_id, tokenize_func, sp_processor
    
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}. Choose from: char, word, sentencepiece")