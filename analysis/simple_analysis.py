#!/usr/bin/env python3
"""
Simple dataset analyzer without loading large files
"""

import os

def analyze_datasets_simple():
    """Simple analysis without loading huge files"""
    
    text_data_dir = '/home/akshat/GPT_from_scratch/text_data'
    
    datasets = {
        'alice_story.txt': {'desc': 'Very small test dataset', 'use': 'Quick testing'},
        'alice_extended.txt': {'desc': 'Small test dataset', 'use': 'Basic testing'},
        'pg76702.txt': {'desc': 'Medium Project Gutenberg', 'use': 'Small-scale training'},
        'jane_austen_clean.txt': {'desc': 'Classic literature (clean)', 'use': 'Medium training'},
        'wikitext_full.txt': {'desc': 'Large Wikipedia dataset', 'use': 'Large training'},
        'BookCorpus3_cleaned.txt': {'desc': 'Very large book corpus', 'use': 'Production training'}
    }
    
    print("=== DATASET ANALYSIS ===\n")
    
    for filename, info in datasets.items():
        filepath = os.path.join(text_data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            print(f"📁 {filename}")
            print(f"   Description: {info['desc']}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Best for: {info['use']}")
            
            # Estimate tokens (rough)
            if size_mb < 1:
                tokens_est = int(size_mb * 1024 * 1024 * 0.8)  # ~0.8 tokens per byte for small files
            else:
                tokens_est = int(size_mb * 1024 * 1024 * 0.3)  # ~0.3 tokens per byte for larger files
            
            print(f"   Estimated tokens: {tokens_est:,}")
            
            # Training time estimate
            if size_mb < 1:
                time_est = "5-15 minutes"
            elif size_mb < 10:
                time_est = "30-60 minutes"
            elif size_mb < 100:
                time_est = "2-4 hours"
            elif size_mb < 1000:
                time_est = "6-12 hours"
            else:
                time_est = "12-48 hours"
            
            print(f"   Training time: {time_est}")
            print()

def model_config_analysis():
    """Analyze model configurations"""
    
    configs = {
        'config_char.txt': {
            'tokenizer': 'Character-level',
            'params_est': '2-5M parameters',
            'memory_est': '200-500 MB',
            'good_for': 'Small datasets, fast training'
        },
        'config_word.txt': {
            'tokenizer': 'Word-level',
            'params_est': '15-30M parameters', 
            'memory_est': '1-2 GB',
            'good_for': 'Medium datasets, balanced approach'
        },
        'config_sentencepiece.txt': {
            'tokenizer': 'SentencePiece',
            'params_est': '50-100M parameters',
            'memory_est': '4-6 GB', 
            'good_for': 'Large datasets, production quality'
        }
    }
    
    print("=== MODEL CONFIGURATION ANALYSIS ===\n")
    
    for config_file, info in configs.items():
        print(f"⚙️ {config_file}")
        print(f"   Tokenizer: {info['tokenizer']}")
        print(f"   Est. parameters: {info['params_est']}")
        print(f"   Est. GPU memory: {info['memory_est']}")
        print(f"   Best for: {info['good_for']}")
        print()

def recommendations():
    """Provide specific recommendations"""
    
    print("=== MY RECOMMENDATIONS FOR YOU ===\n")
    
    print("🎯 **RECOMMENDED TRAINING PATH:**\n")
    
    print("1️⃣ **START HERE (Recommended)**")
    print("   📄 Dataset: jane_austen_clean.txt (4.1 MB)")
    print("   ⚙️ Config: config_char.txt")
    print("   ⏱️ Time: ~30-60 minutes")
    print("   🎯 Goal: Learn how training works, get first working model")
    print("   💾 Memory: ~200 MB (safe for your 8GB GPU)")
    print()
    
    print("2️⃣ **INTERMEDIATE STEP**")
    print("   📄 Dataset: jane_austen_clean.txt (4.1 MB)")
    print("   ⚙️ Config: config_word.txt")
    print("   ⏱️ Time: ~1-2 hours")
    print("   🎯 Goal: Try word-level tokenization, larger model")
    print("   💾 Memory: ~1.5 GB")
    print()
    
    print("3️⃣ **SCALE UP**")
    print("   📄 Dataset: wikitext_full.txt (513 MB)")
    print("   ⚙️ Config: config_word.txt")
    print("   ⏱️ Time: ~4-8 hours")
    print("   🎯 Goal: Train on diverse, larger dataset")
    print("   💾 Memory: ~2-3 GB")
    print()
    
    print("4️⃣ **ADVANCED (Weekend Project)**")
    print("   📄 Dataset: BookCorpus3_cleaned.txt (2.1 GB)")
    print("   ⚙️ Config: config_sentencepiece.txt")
    print("   ⏱️ Time: ~12-24 hours")
    print("   🎯 Goal: Production-quality model")
    print("   💾 Memory: ~5-6 GB")
    print()
    
    print("⚡ **QUICK START COMMAND:**")
    print("   1. Copy config_char.txt to config.txt")
    print("   2. Edit TEXT_FILE_PATH to point to jane_austen_clean.txt")
    print("   3. Run: python train.py")
    print("   4. Watch your first GPT model train! 🚀")
    print()
    
    print("🔧 **WHY JANE AUSTEN + CHARACTER CONFIG?**")
    print("   ✅ Clean, high-quality text")
    print("   ✅ Perfect size for learning (not too big/small)")
    print("   ✅ Character tokenization is simple and reliable")
    print("   ✅ Fast training on your hardware")
    print("   ✅ Will produce a working text generator")
    print()
    
    print("⚠️ **AVOID THESE MISTAKES:**")
    print("   ❌ Don't start with BookCorpus (too big for first run)")
    print("   ❌ Don't use SentencePiece config on small datasets")
    print("   ❌ Don't train for too many epochs initially")
    print("   ❌ Don't ignore GPU memory warnings")

if __name__ == "__main__":
    analyze_datasets_simple()
    model_config_analysis()
    recommendations()
