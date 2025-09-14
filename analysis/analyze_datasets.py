#!/usr/bin/env python3
"""
Dataset and configuration analyzer
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import load_config, create_tokenizer

def analyze_datasets():
    """Analyze all available datasets"""
    
    text_data_dir = '/home/akshat/GPT_from_scratch/text_data'
    
    datasets = {
        'alice_story.txt': 'Very small test dataset',
        'alice_extended.txt': 'Small test dataset', 
        'dummy.txt': 'Tiny test dataset',
        'pg76702.txt': 'Medium Project Gutenberg text',
        'jane_austen_clean.txt': 'Medium classical literature',
        'wikitext_full.txt': 'Large Wikipedia dataset',
        'BookCorpus3_cleaned.txt': 'Very large book corpus'
    }
    
    print("=== DATASET ANALYSIS ===\n")
    
    for filename, description in datasets.items():
        filepath = os.path.join(text_data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            # Read sample to check quality
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    sample = f.read(200)
                
                # Count characters and estimate tokens
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                char_count = len(content)
                word_count = len(content.split())
                
                print(f"📁 {filename}")
                print(f"   Description: {description}")
                print(f"   Size: {size_mb:.1f} MB")
                print(f"   Characters: {char_count:,}")
                print(f"   Words: {word_count:,}")
                print(f"   Sample: '{sample[:100]}...'")
                print()
                
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                print()

def estimate_model_sizes():
    """Estimate model parameter counts and memory requirements"""
    
    configs = {
        'Character-level': '/home/akshat/GPT_from_scratch/cleaned_code/config_char.txt',
        'Word-level': '/home/akshat/GPT_from_scratch/cleaned_code/config_word.txt', 
        'SentencePiece': '/home/akshat/GPT_from_scratch/cleaned_code/config_sentencepiece.txt'
    }
    
    print("=== MODEL SIZE ESTIMATES ===\n")
    
    for name, config_path in configs.items():
        config = load_config(config_path)
        
        # Estimate parameters
        d_model = config['D_MODEL']
        vocab_size = config.get('VOCAB_SIZE', 100)  # Fallback for char
        context_len = config['CONTEXT_LEN']
        n_heads = config['ATTENTION_HEADS']
        n_blocks = config['DECODER_BLOCKS']
        
        # Parameter estimation (rough)
        # Embedding: vocab_size * d_model + context_len * d_model
        embedding_params = vocab_size * d_model + context_len * d_model
        
        # Each transformer block: ~12 * d_model^2 (rough estimate)
        transformer_params = n_blocks * 12 * d_model * d_model
        
        # Output layer: d_model * vocab_size
        output_params = d_model * vocab_size
        
        total_params = embedding_params + transformer_params + output_params
        
        # Memory estimation (rough)
        # Parameters: 4 bytes each (float32)
        # Activations: batch_size * context_len * d_model * n_blocks * 4 bytes
        param_memory_mb = (total_params * 4) / (1024 * 1024)
        
        # Estimate activation memory for batch_size=8
        activation_memory_mb = (8 * context_len * d_model * n_blocks * 4) / (1024 * 1024)
        
        total_memory_mb = param_memory_mb + activation_memory_mb
        
        print(f"🔧 {name} Configuration:")
        print(f"   Model size: {d_model}d, {n_blocks} blocks, {n_heads} heads")
        print(f"   Context length: {context_len}")
        print(f"   Vocabulary: {vocab_size}")
        print(f"   Estimated parameters: {total_params/1_000_000:.1f}M")
        print(f"   Estimated GPU memory: {total_memory_mb:.0f} MB")
        print(f"   Training epochs: {config['EPOCHS']}")
        print()

def recommend_configuration():
    """Provide recommendations based on analysis"""
    
    print("=== RECOMMENDATIONS ===\n")
    
    print("🎯 **RECOMMENDED TRAINING PROGRESSION:**\n")
    
    print("1️⃣ **BEGINNER (Start Here)**")
    print("   Dataset: jane_austen_clean.txt (4.2MB)")
    print("   Config: config_char.txt")
    print("   Why: Small, clean dataset with good literary quality")
    print("   Expected training time: 30-60 minutes")
    print("   GPU memory: ~200MB")
    print()
    
    print("2️⃣ **INTERMEDIATE**") 
    print("   Dataset: wikitext_full.txt (514MB)")
    print("   Config: config_word.txt")
    print("   Why: Larger dataset with diverse content")
    print("   Expected training time: 3-6 hours")
    print("   GPU memory: ~2GB")
    print()
    
    print("3️⃣ **ADVANCED**")
    print("   Dataset: BookCorpus3_cleaned.txt (2.1GB)")
    print("   Config: config_sentencepiece.txt")
    print("   Why: Large-scale training with modern tokenization")
    print("   Expected training time: 12-24 hours")
    print("   GPU memory: ~6GB")
    print()
    
    print("⚠️ **HARDWARE CONSIDERATIONS:**")
    print("   Your RTX 4060 Laptop has 8GB VRAM")
    print("   ✅ All configurations should fit")
    print("   ✅ Start with jane_austen + char config")
    print("   ✅ Monitor GPU memory during training")
    print()
    
    print("🚀 **QUICK START RECOMMENDATION:**")
    print("   Use: jane_austen_clean.txt + config_char.txt")
    print("   Why: Perfect balance of quality and training time")
    print("   This will give you a working GPT model in ~1 hour!")

if __name__ == "__main__":
    analyze_datasets()
    estimate_model_sizes()
    recommend_configuration()
