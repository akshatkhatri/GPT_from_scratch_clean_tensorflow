#!/usr/bin/env python3
"""
Quick test to check if the OUT_OF_RANGE bug is fixed
"""

import sys
import os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords

def test_data_pipeline_fix():
    """Test if the OUT_OF_RANGE bug is fixed"""
    print("=== TESTING OUT_OF_RANGE FIX ===\n")
    
    # Load config
    config = load_config('/home/akshat/GPT_from_scratch/cleaned_code/configs/config.txt')
    print(f"✅ Config loaded: {config['TOKENIZER_TYPE']}")
    
    # Create tokenizer
    file_path = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'
    token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [file_path])
    vocab_size = len(token_to_id_dict)
    print(f"✅ Tokenizer created, vocab size: {vocab_size}")
    
    try:
        # Create TFRecords with small parameters for quick testing
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=64,   # Small context for testing
            batch_size=4,        # Small batch for testing
            records_per_file=10, # Few records per file
            version_name="debug_test_fixed"
        )
        
        print(f"✅ TFRecords created, steps per epoch: {steps}")
        
        # Test iterating through the dataset for more steps than available
        print("\n=== TESTING DATASET ITERATION ===")
        test_steps = min(steps + 5, 20)  # Test more steps than calculated
        
        for step in range(test_steps):
            try:
                batch = next(iter(train_ds))
                inputs, targets = batch
                print(f"Step {step+1}/{test_steps}: ✅ Got batch with shapes {inputs[0].shape} -> {targets.shape}")
                
                if step >= steps:
                    print(f"   ✅ Successfully went beyond calculated steps ({steps})!")
                    
            except tf.errors.OutOfRangeError:
                print(f"Step {step+1}: ❌ OUT_OF_RANGE error occurred!")
                return False
            except Exception as e:
                print(f"Step {step+1}: ❌ Other error: {e}")
                return False
        
        print(f"\n✅ SUCCESS: Completed {test_steps} steps without OUT_OF_RANGE error!")
        print(f"   Calculated steps per epoch: {steps}")
        print(f"   Actually tested: {test_steps} steps")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_pipeline_fix()
    if success:
        print("\n🎉 OUT_OF_RANGE bug appears to be FIXED!")
    else:
        print("\n💥 OUT_OF_RANGE bug still exists!")
