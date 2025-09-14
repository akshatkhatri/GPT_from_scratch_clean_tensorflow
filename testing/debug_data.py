#!/usr/bin/env python3
"""
Debug script to identify bugs in data processing pipeline
"""

import sys
import os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT

def debug_data_pipeline():
    """Debug the data processing pipeline step by step"""
    print("=== DEBUGGING DATA PIPELINE ===\n")
    
    # Load config
    config = load_config('/home/akshat/GPT_from_scratch/cleaned_code/configs/config.txt')
    print(f"✅ Config loaded: {config['TOKENIZER_TYPE']}")
    
    # Create tokenizer
    file_path = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'
    token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [file_path])
    vocab_size = len(token_to_id_dict)
    print(f"✅ Tokenizer created, vocab size: {vocab_size}")
    
    # Test small sample text processing
    sample_text = "Hello world! This is a test."
    sample_tokens = tokenize_func(sample_text)
    print(f"✅ Sample tokenization: '{sample_text}' -> {sample_tokens}")
    
    try:
        # Create TFRecords with very small parameters for testing
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=32,  # Small context for testing
            batch_size=2,       # Small batch for testing
            records_per_file=5, # Few records per file
            version_name="debug_test"
        )
        
        print(f"✅ TFRecords created, steps per epoch: {steps}")
        
        # Test data shapes from dataset
        print("\n=== TESTING DATA SHAPES ===")
        for batch_idx, (inputs, targets) in enumerate(train_ds.take(1)):
            print(f"Batch {batch_idx}:")
            print(f"  Inputs type: {type(inputs)}")
            print(f"  Targets type: {type(targets)}")
            
            if isinstance(inputs, tuple):
                input_ids, attention_mask = inputs
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Attention mask shape: {attention_mask.shape}")
                print(f"  Input IDs dtype: {input_ids.dtype}")
                print(f"  Attention mask dtype: {attention_mask.dtype}")
            else:
                print(f"  Single input shape: {inputs.shape}")
                print(f"  Input dtype: {inputs.dtype}")
            
            print(f"  Targets shape: {targets.shape}")
            print(f"  Targets dtype: {targets.dtype}")
            
            # Check sample values
            print(f"  Sample input IDs: {input_ids[0][:10]}...")
            print(f"  Sample targets: {targets[0][:10]}...")
            print(f"  Sample attention mask: {attention_mask[0][:10]}...")
        
        # Test model creation and prediction
        print("\n=== TESTING MODEL COMPATIBILITY ===")
        model = GPT(
            d_model=config['D_MODEL'],
            vocab_size=vocab_size,
            context_length=32,  # Match the test context length
            attention_heads=config['ATTENTION_HEADS'],
            epsilon=config['LEARNING_RATE'],
            decoder_blocks=config['DECODER_BLOCKS'],
            dropout_rate=config['DROPOUT_RATE']
        )
        
        print(f"✅ GPT model created")
        
        # Test model prediction on one batch
        for inputs, targets in train_ds.take(1):
            try:
                output = model(inputs, training=False)
                print(f"✅ Model prediction successful")
                print(f"  Output shape: {output.shape}")
                print(f"  Expected shape: (batch_size, seq_len, vocab_size)")
                print(f"  Actual: ({output.shape[0]}, {output.shape[1]}, {output.shape[2]})")
                
                # Check if shapes match expectations
                batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
                seq_len = inputs[0].shape[1] if isinstance(inputs, tuple) else inputs.shape[1]
                
                if output.shape == (batch_size, seq_len, vocab_size):
                    print(f"✅ Output shape is correct!")
                else:
                    print(f"❌ Output shape mismatch!")
                    
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                import traceback
                traceback.print_exc()
                
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data_pipeline()
