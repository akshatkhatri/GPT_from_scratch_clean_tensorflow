#!/usr/bin/env python3
"""
Test all three tokenizer types to ensure they work correctly
"""

import sys
import os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords

def test_tokenizer_type(tokenizer_type):
    """Test a specific tokenizer type"""
    print(f"\n{'='*60}")
    print(f"TESTING {tokenizer_type.upper()} TOKENIZER")
    print(f"{'='*60}")
    
    # Update config for this tokenizer type
    config_path = '/home/akshat/GPT_from_scratch/cleaned_code/configs/config.txt'
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Update TOKENIZER_TYPE
    new_lines = []
    for line in lines:
        if line.startswith('TOKENIZER_TYPE='):
            new_lines.append(f'TOKENIZER_TYPE={tokenizer_type}\n')
        else:
            new_lines.append(line)
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    try:
        # Load config
        config = load_config(config_path)
        print(f"✅ Config updated: {config['TOKENIZER_TYPE']}")
        
        # Create tokenizer
        file_path = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'
        token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [file_path])
        vocab_size = len(token_to_id_dict)
        print(f"✅ Tokenizer created, vocab size: {vocab_size}")
        
        # Test sample
        sample_text = "Hello world! This is a test. How are you?"
        sample_tokens = tokenize_func(sample_text)
        print(f"✅ Sample: '{sample_text}' -> {len(sample_tokens)} tokens")
        print(f"   Token IDs: {sample_tokens[:10]}...")
        
        # Test edge cases
        empty_tokens = tokenize_func("")
        print(f"✅ Empty string: -> {len(empty_tokens)} tokens")
        
        single_char_tokens = tokenize_func("a")
        print(f"✅ Single char: 'a' -> {len(single_char_tokens)} tokens")
        
        # Test vocabulary size limits
        if tokenizer_type in ['word', 'sentencepiece']:
            expected_vocab_size = config.get('VOCAB_SIZE', 50257)
            if vocab_size > expected_vocab_size:
                print(f"⚠️  WARNING: Vocab size {vocab_size} exceeds config limit {expected_vocab_size}")
            else:
                print(f"✅ Vocab size {vocab_size} within limit {expected_vocab_size}")
        
        # Quick TFRecord test
        print("Testing TFRecord creation...")
        train_ds, val_ds, steps = prepare_tfrecords(
            text_file_path=file_path,
            token_to_id_dict=token_to_id_dict,
            tokenize_func=tokenize_func,
            context_length=32,   # Small for quick test
            batch_size=2,        # Small for quick test
            records_per_file=50,
            version_name=f"test_{tokenizer_type}"
        )
        
        print(f"✅ TFRecords created, steps: {steps}")
        
        # Test one batch
        for inputs, targets in train_ds.take(1):
            input_ids, attention_mask = inputs
            print(f"✅ Batch test - Input shape: {input_ids.shape}, Targets shape: {targets.shape}")
            
            # Check value ranges
            max_input = tf.reduce_max(input_ids)
            max_target = tf.reduce_max(targets)
            if max_input >= vocab_size or max_target >= vocab_size:
                print(f"❌ ERROR: Token IDs exceed vocab size!")
                return False
            else:
                print(f"✅ Token IDs within vocab range [0, {vocab_size-1}]")
        
        # Cleanup
        import shutil
        test_dir = f'./tfrecords/test_{tokenizer_type}'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        print(f"✅ {tokenizer_type.upper()} TOKENIZER TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {tokenizer_type.upper()} TOKENIZER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all tokenizer types"""
    print("TESTING ALL TOKENIZER TYPES")
    print("="*60)
    
    results = {}
    tokenizer_types = ['char', 'word', 'sentencepiece']
    
    for tokenizer_type in tokenizer_types:
        results[tokenizer_type] = test_tokenizer_type(tokenizer_type)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    all_passed = True
    for tokenizer_type, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{tokenizer_type.upper()}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TOKENIZER TYPES WORK CORRECTLY!")
    else:
        print("\n💥 SOME TOKENIZER TYPES HAVE ISSUES!")
    
    # Reset config to char for safety
    config_path = '/home/akshat/GPT_from_scratch/cleaned_code/configs/config.txt'
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith('TOKENIZER_TYPE='):
            new_lines.append('TOKENIZER_TYPE=char\n')
        else:
            new_lines.append(line)
    
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print("🔄 Reset tokenizer type to 'char'")

if __name__ == "__main__":
    main()
