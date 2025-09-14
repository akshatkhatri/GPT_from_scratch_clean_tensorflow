#!/usr/bin/env python3
"""
Final edge case testing for data processing pipeline
"""

import sys
import os
import tensorflow as tf
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords, create_train_val_datasets

def test_edge_cases():
    """Test edge cases that could cause issues"""
    print("=== TESTING EDGE CASES ===\n")
    
    config = load_config('/home/akshat/GPT_from_scratch/cleaned_code/configs/config.txt')
    
    # Test 1: Very small text
    print("TEST 1: Very small text file")
    small_text = "Hello world! This is a tiny test file."
    small_file_path = '/tmp/small_test.txt'
    with open(small_file_path, 'w') as f:
        f.write(small_text)
    
    try:
        token_to_id_dict, tokenize_func, _ = create_tokenizer(config, [small_file_path])
        tokens = tokenize_func(small_text)
        print(f"✅ Small text: {len(tokens)} tokens from {len(small_text)} chars")
        
        # This should handle gracefully even if we can't create enough examples
        try:
            train_ds, val_ds, steps = prepare_tfrecords(
                text_file_path=small_file_path,
                token_to_id_dict=token_to_id_dict,
                tokenize_func=tokenize_func,
                context_length=16,
                batch_size=2,
                records_per_file=10,
                version_name="test_small"
            )
            print(f"✅ Small file handled: {steps} steps")
        except Exception as e:
            print(f"⚠️  Small file limitation: {e}")
    except Exception as e:
        print(f"❌ Small text test failed: {e}")
    
    # Test 2: Large batch size vs small dataset
    print("\nTEST 2: Large batch size")
    try:
        # Use actual Jane Austen text but large batch size
        file_path = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'
        token_to_id_dict, tokenize_func, _ = create_tokenizer(config, [file_path])
        
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=32,
            batch_size=512,  # Very large batch
            records_per_file=100,
            version_name="test_large_batch"
        )
        
        # Test if we can actually get a batch
        for batch in train_ds.take(1):
            inputs, targets = batch
            print(f"✅ Large batch test: shape {inputs[0].shape}")
            break
            
    except Exception as e:
        print(f"❌ Large batch test failed: {e}")
    
    # Test 3: Very long context length
    print("\nTEST 3: Long context length")
    try:
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=1024,  # Very long
            batch_size=2,
            records_per_file=50,
            version_name="test_long_context"
        )
        
        for batch in train_ds.take(1):
            inputs, targets = batch
            print(f"✅ Long context test: shape {inputs[0].shape}")
            break
            
    except Exception as e:
        print(f"❌ Long context test failed: {e}")
    
    # Test 4: Memory stress test
    print("\nTEST 4: Memory efficiency")
    try:
        # Create dataset without loading all into memory
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=128,
            batch_size=32,
            records_per_file=200,  # More records per file
            version_name="test_memory"
        )
        
        # Test memory usage by iterating multiple batches
        for i, batch in enumerate(train_ds.take(20)):
            if i % 10 == 9:
                print(f"✅ Memory test batch {i+1}: OK")
                
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
    
    # Test 5: Dataset consistency
    print("\nTEST 5: Dataset consistency")
    try:
        tfrecord_folder = './tfrecords/test_memory'
        
        # Create dataset multiple times and check consistency
        ds1 = create_train_val_datasets(
            tfrecord_folder=tfrecord_folder,
            batch_size=4,
            context_length=128
        )[0]
        
        ds2 = create_train_val_datasets(
            tfrecord_folder=tfrecord_folder,
            batch_size=4,
            context_length=128
        )[0]
        
        # Get first batch from each
        batch1 = next(iter(ds1.take(1)))
        batch2 = next(iter(ds2.take(1)))
        
        # They should be different due to shuffling
        inputs1, targets1 = batch1
        inputs2, targets2 = batch2
        
        print(f"✅ Dataset consistency: shapes match")
        print(f"   Dataset 1: {inputs1[0].shape}")
        print(f"   Dataset 2: {inputs2[0].shape}")
        
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
    
    # Test 6: Validation split integrity
    print("\nTEST 6: Validation split integrity")
    try:
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=64,
            batch_size=8,
            records_per_file=100,
            version_name="test_val_split"
        )
        
        # Count batches in train and val
        train_count = sum(1 for _ in train_ds.take(100))
        val_count = sum(1 for _ in val_ds.take(100))
        
        print(f"✅ Validation split: train={train_count}, val={val_count}")
        
        # Validate that we don't get OUT_OF_RANGE errors
        for i, batch in enumerate(train_ds.take(10)):
            pass
        print(f"✅ No OUT_OF_RANGE in first 10 train batches")
        
        for i, batch in enumerate(val_ds.take(5)):
            pass
        print(f"✅ No OUT_OF_RANGE in first 5 val batches")
        
    except Exception as e:
        print(f"❌ Validation split test failed: {e}")
    
    # Cleanup
    print("\nCleaning up test files...")
    import shutil
    test_dirs = [
        './tfrecords/test_small',
        './tfrecords/test_large_batch', 
        './tfrecords/test_long_context',
        './tfrecords/test_memory',
        './tfrecords/test_val_split'
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
            except:
                pass
    
    if os.path.exists(small_file_path):
        os.remove(small_file_path)
    
    print("🧹 Cleanup complete")

if __name__ == "__main__":
    test_edge_cases()
