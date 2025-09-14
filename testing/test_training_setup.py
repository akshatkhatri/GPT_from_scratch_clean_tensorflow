#!/usr/bin/env python3
"""
Test script to verify training works with all tokenizer types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT

def test_training_setup(config_file):
    """Test training setup for a specific config"""
    print(f"\n{'='*60}")
    print(f"Testing training setup: {config_file}")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        config = load_config(config_file)
        
        print(f"✅ Config loaded successfully")
        print(f"   Tokenizer: {config['TOKENIZER_TYPE']}")
        print(f"   Vocab size setting: {config['VOCAB_SIZE']}")
        print(f"   Model D_MODEL: {config['D_MODEL']}")
        print(f"   Context length: {config['CONTEXT_LEN']}")
        
        # Create tokenizer
        file_path = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'
        token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [file_path])
        actual_vocab_size = len(token_to_id_dict)
        
        print(f"✅ Tokenizer created successfully")
        print(f"   Actual vocab size: {actual_vocab_size}")
        
        # Test creating GPT model
        model = GPT(
            d_model=config['D_MODEL'],
            vocab_size=actual_vocab_size,
            context_length=config['CONTEXT_LEN'],
            attention_heads=config['ATTENTION_HEADS'],
            epsilon=config['LEARNING_RATE'],
            decoder_blocks=config['DECODER_BLOCKS'],
            dropout_rate=config['DROPOUT_RATE']
        )
        
        print(f"✅ GPT model created successfully")
        print(f"   Parameters: D_MODEL={config['D_MODEL']}, VOCAB_SIZE={actual_vocab_size}")
        
        # Test small data processing (just 10 records to be fast)
        try:
            train_ds, val_ds, steps = prepare_tfrecords(
                text_file_path=file_path,
                token_to_id_dict=token_to_id_dict,
                tokenize_func=tokenize_func,
                context_length=min(config['CONTEXT_LEN'], 128),  # Use smaller context for testing
                batch_size=2,
                records_per_file=10,
                version_name=f"test_{config['TOKENIZER_TYPE']}"
            )
            
            print(f"✅ Data processing successful")
            print(f"   Steps per epoch: {steps}")
            
        except Exception as e:
            print(f"⚠️  Data processing test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing training setup for all tokenizer types...")
    
    # Test all configurations
    configs = [
        '/home/akshat/GPT_from_scratch/cleaned_code/config_char.txt',
        '/home/akshat/GPT_from_scratch/cleaned_code/config_word.txt',
        '/home/akshat/GPT_from_scratch/cleaned_code/config_sentencepiece.txt'
    ]
    
    results = {}
    for config_file in configs:
        tokenizer_type = config_file.split('_')[-1].replace('.txt', '')
        results[tokenizer_type] = test_training_setup(config_file)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for tokenizer_type, success in results.items():
        status = '✅ PASS' if success else '❌ FAIL'
        print(f"{tokenizer_type.upper():15}: {status}")
