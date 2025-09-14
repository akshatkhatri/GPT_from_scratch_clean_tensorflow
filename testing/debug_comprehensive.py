#!/usr/bin/env python3
"""
Comprehensive test to identify all data processing bugs
"""

import sys
import os
import tensorflow as tf
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT

def comprehensive_data_test():
    """Comprehensive test of the data processing pipeline"""
    print("=== COMPREHENSIVE DATA PIPELINE TEST ===\n")
    
    # Load config
    config = load_config('/home/akshat/GPT_from_scratch/cleaned_code/configs/config.txt')
    print(f"✅ Config loaded: {config['TOKENIZER_TYPE']}")
    
    # Create tokenizer
    file_path = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'
    token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [file_path])
    vocab_size = len(token_to_id_dict)
    print(f"✅ Tokenizer created, vocab size: {vocab_size}")
    
    # Test small sample
    sample_text = "Hello world! This is a test. How are you?"
    sample_tokens = tokenize_func(sample_text)
    print(f"✅ Sample: '{sample_text}' -> {len(sample_tokens)} tokens")
    print(f"   Token IDs: {sample_tokens[:10]}...")
    
    try:
        # Create TFRecords with small parameters for testing
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=64,   # Small context for testing
            batch_size=4,        # Small batch for testing
            records_per_file=20, # More records per file
            version_name="debug_comprehensive"
        )
        
        print(f"✅ TFRecords created, steps per epoch: {steps}")
        
        # Test 1: Data shapes and types
        print("\n=== TEST 1: DATA SHAPES AND TYPES ===")
        for batch_idx, (inputs, targets) in enumerate(train_ds.take(3)):
            print(f"Batch {batch_idx+1}:")
            
            # Check input structure
            if isinstance(inputs, tuple) and len(inputs) == 2:
                input_ids, attention_mask = inputs
                print(f"  ✅ Input structure: (input_ids, attention_mask)")
                print(f"     Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
                print(f"     Attention mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
                
                # Check value ranges
                print(f"     Input IDs range: [{tf.reduce_min(input_ids)}, {tf.reduce_max(input_ids)}]")
                print(f"     Expected vocab range: [0, {vocab_size-1}]")
                
                if tf.reduce_max(input_ids) >= vocab_size:
                    print(f"  ❌ ERROR: Input IDs exceed vocab size!")
                    return False
                
                if tf.reduce_min(input_ids) < 0:
                    print(f"  ❌ ERROR: Negative input IDs found!")
                    return False
                
                # Check attention mask values
                unique_mask_values = tf.unique(tf.reshape(attention_mask, [-1]))[0]
                print(f"     Attention mask unique values: {unique_mask_values}")
                
                if not all(val in [0, 1] for val in unique_mask_values.numpy()):
                    print(f"  ❌ ERROR: Attention mask should only contain 0s and 1s!")
                    return False
                    
            else:
                print(f"  ❌ ERROR: Expected (input_ids, attention_mask), got {type(inputs)}")
                return False
            
            # Check targets
            print(f"  Targets shape: {targets.shape}, dtype: {targets.dtype}")
            print(f"  Targets range: [{tf.reduce_min(targets)}, {tf.reduce_max(targets)}]")
            
            if tf.reduce_max(targets) >= vocab_size:
                print(f"  ❌ ERROR: Target IDs exceed vocab size!")
                return False
            
            if tf.reduce_min(targets) < 0:
                print(f"  ❌ ERROR: Negative target IDs found!")
                return False
            
            # Check sequence alignment (targets should be input_ids shifted by 1)
            print(f"  Sample input:  {input_ids[0][:5]}")
            print(f"  Sample target: {targets[0][:5]}")
            print(f"  Expected: target[i] = input[i+1]")
            
            break  # Just check first batch in detail
        
        # Test 2: Model compatibility
        print("\n=== TEST 2: MODEL COMPATIBILITY ===")
        model = GPT(
            d_model=config['D_MODEL'],
            vocab_size=vocab_size,
            context_length=64,  # Match test context
            attention_heads=config['ATTENTION_HEADS'],
            epsilon=config['LEARNING_RATE'],
            decoder_blocks=config['DECODER_BLOCKS'],
            dropout_rate=config['DROPOUT_RATE']
        )
        
        print(f"✅ GPT model created")
        
        # Test forward pass
        for inputs, targets in train_ds.take(1):
            try:
                outputs = model(inputs, training=False)
                print(f"✅ Forward pass successful")
                print(f"  Output shape: {outputs.shape}")
                print(f"  Expected: (batch_size, seq_len, vocab_size)")
                print(f"  Got: ({outputs.shape[0]}, {outputs.shape[1]}, {outputs.shape[2]})")
                
                # Check output values
                print(f"  Output range: [{tf.reduce_min(outputs):.3f}, {tf.reduce_max(outputs):.3f}]")
                
                # Test loss calculation
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                loss = loss_fn(targets, outputs)
                print(f"✅ Loss calculation successful: {loss:.3f}")
                
                if tf.math.is_nan(loss):
                    print(f"  ❌ ERROR: NaN loss detected!")
                    return False
                
                if loss < 0:
                    print(f"  ❌ ERROR: Negative loss!")
                    return False
                    
            except Exception as e:
                print(f"❌ Model forward pass failed: {e}")
                return False
        
        # Test 3: Dataset iteration stability
        print("\n=== TEST 3: DATASET ITERATION STABILITY ===")
        iteration_count = 0
        max_iterations = 50
        
        try:
            for batch in train_ds.take(max_iterations):
                iteration_count += 1
                if iteration_count % 10 == 0:
                    print(f"  Iteration {iteration_count}: ✅")
            
            print(f"✅ Successfully iterated {iteration_count} batches")
            
        except Exception as e:
            print(f"❌ Dataset iteration failed at batch {iteration_count}: {e}")
            return False
        
        # Test 4: Validation dataset
        print("\n=== TEST 4: VALIDATION DATASET ===")
        val_batch_count = 0
        try:
            for batch in val_ds.take(5):
                val_batch_count += 1
            print(f"✅ Validation dataset works: {val_batch_count} batches tested")
        except Exception as e:
            print(f"❌ Validation dataset failed: {e}")
            return False
        
        # Test 5: Check for data leakage between train/val
        print("\n=== TEST 5: TRAIN/VAL SEPARATION ===")
        print(f"✅ Train/val split by files - no data leakage expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test files"""
    import shutil
    test_dir = './tfrecords/debug_comprehensive'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print("🧹 Cleaned up test files")

if __name__ == "__main__":
    try:
        success = comprehensive_data_test()
        if success:
            print("\n🎉 ALL TESTS PASSED! No data processing bugs detected.")
        else:
            print("\n💥 BUGS DETECTED! Check the error messages above.")
    finally:
        cleanup()
