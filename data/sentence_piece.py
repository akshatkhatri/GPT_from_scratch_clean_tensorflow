#!/usr/bin/env python3
import os
import sys
from typing import List, Dict, Tuple
import numpy as np
import tensorflow as tf
import keras
import sentencepiece as spm
import time

def train_sentencepiece_tokenizer(file_path_list: List[str], 
                                vocab_size: int = 2000,
                                model_prefix: str = 'spm_gpt',
                                input_sentence_size=2000000,
                                shuffle_input_sentence=True) -> spm.SentencePieceProcessor:
    """
    Train SentencePiece tokenizer from text files.
    """
    if isinstance(file_path_list, (str, bytes)):
        file_path_list = [file_path_list]
    
    # Validate files
    for file_name in file_path_list:
        if os.path.isdir(file_name):
            raise IsADirectoryError(f"Expected file path, got directory: {file_name}")
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
    
    # Show file info
    for file_path in file_path_list:
        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"Processing file: {file_path} ({size_mb:.1f} MB)")
    
    input_files = ','.join(file_path_list)
    
    print(f"Starting SentencePiece training with {vocab_size} vocab size...")
    print("Training progress (this may take 5-15 minutes)...")
    
    start_time = time.time()
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        num_threads=8,
        input_sentence_size=2000000,
        shuffle_input_sentence=True
        
    )
    
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.1f} seconds!")
    print(f"Model saved as {model_prefix}.model")
    
    # Load and return processor
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    
    print(f"Actual vocabulary size: {sp.get_piece_size()}")
    return sp

def tokenize_and_build_token_id_sp(sp: spm.SentencePieceProcessor, 
                                 text_batch: List[str], 
                                 max_seq_len: int, 
                                 pad_value: int = 0) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Tokenize batch of text using SentencePiece.
    """
    batch_token_ids = []
    
    for text in text_batch:
        ids = sp.encode_as_ids(text)
        
        if len(ids) > max_seq_len:
            ids = ids[-max_seq_len:]
        else:
            ids += [pad_value] * (max_seq_len - len(ids))
        
        batch_token_ids.append(ids)
    
    token_ids = np.array(batch_token_ids, dtype=np.int32)
    attention_mask = (token_ids != pad_value).astype(np.int32)
    
    return tf.constant(token_ids), tf.constant(attention_mask)

if __name__ == "__main__":
    # Configuration
    VOCAB_SIZE = 8000
    CONTEXT_LEN = 512
    
    print("=== SentencePiece Training Test ===")
    print(f"Target vocab size: {VOCAB_SIZE}")
    print(f"Context length: {CONTEXT_LEN}")
    print()
    
    try:
        # Train tokenizer
        print("Step 1: Training SentencePiece tokenizer...")
        sp = train_sentencepiece_tokenizer(
            [r'/home/akshat/GPT_from_scratch/text_data/BookCorpus3_cleaned.txt'], 
            vocab_size=VOCAB_SIZE
        )
        
        # Update vocab size to actual size
        actual_vocab_size = sp.get_piece_size()
        print(f"Updated VOCAB_SIZE from {VOCAB_SIZE} to {actual_vocab_size}")
        
        print("\nStep 2: Testing tokenization...")
        batch_text = ['yo', 'Akshat Khatri', 'Hello World', 'Me']
        token_ids, attention_mask = tokenize_and_build_token_id_sp(sp, batch_text, CONTEXT_LEN)
        
        print("✅ Tokenization successful!")
        print(f"Token IDs shape: {token_ids.shape}")
        print(f"Token IDs:\n{token_ids}")
        
        # Test decoding
        print("\nStep 3: Testing decoding...")
        for i, text in enumerate(batch_text):
            original_text = text
            tokens = sp.encode_as_ids(text)
            decoded_text = sp.decode_ids(tokens)
            print(f"'{original_text}' -> {tokens} -> '{decoded_text}'")
            
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()