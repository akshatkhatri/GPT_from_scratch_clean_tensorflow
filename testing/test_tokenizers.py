#!/usr/bin/env python3
"""
Test script to verify all tokenizer types work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config

def test_tokenizer(tokenizer_type, vocab_size=2000):
    """Test a specific tokenizer type"""
    print(f"\n{'='*50}")
    print(f"Testing {tokenizer_type.upper()} tokenizer")
    print(f"{'='*50}")
    
    # Create test config
    config = {
        'TOKENIZER_TYPE': tokenizer_type,
        'VOCAB_SIZE': vocab_size
    }
    
    # Test file path
    file_path = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'
    
    try:
        # Create tokenizer
        token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [file_path])
        
        print(f"✅ Successfully created {tokenizer_type} tokenizer")
        print(f"   Vocabulary size: {len(token_to_id_dict)}")
        
        # Test tokenization on sample text
        sample_text = "Hello world! This is a test."
        token_ids = tokenize_func(sample_text)
        
        print(f"   Sample text: '{sample_text}'")
        print(f"   Token IDs: {token_ids[:10]}..." if len(token_ids) > 10 else f"   Token IDs: {token_ids}")
        print(f"   Number of tokens: {len(token_ids)}")
        
        # Show some vocab examples
        print("   Sample vocabulary:")
        for i, (token, token_id) in enumerate(list(token_to_id_dict.items())[:10]):
            print(f"     '{token}' -> {token_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create {tokenizer_type} tokenizer: {e}")
        return False

if __name__ == "__main__":
    print("Testing all tokenizer types...")
    
    # Test character-level tokenizer
    success_char = test_tokenizer('char')
    
    # Test word-level tokenizer
    success_word = test_tokenizer('word', vocab_size=1000)
    
    # Test SentencePiece tokenizer (only if sentencepiece is available)
    try:
        import sentencepiece
        success_sp = test_tokenizer('sentencepiece', vocab_size=500)
    except ImportError:
        print(f"\n{'='*50}")
        print("SENTENCEPIECE tokenizer")
        print(f"{'='*50}")
        print("⚠️  SentencePiece not available - install with: pip install sentencepiece")
        success_sp = False
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Character-level:  {'✅ PASS' if success_char else '❌ FAIL'}")
    print(f"Word-level:       {'✅ PASS' if success_word else '❌ FAIL'}")
    print(f"SentencePiece:    {'✅ PASS' if success_sp else '❌ FAIL/UNAVAILABLE'}")
