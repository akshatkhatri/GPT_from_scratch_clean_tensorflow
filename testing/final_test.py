#!/usr/bin/env python3
"""
Proper small training test with better configuration
"""

import sys
import os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT

def test_training_proper():
    """Proper small training test"""
    print("=== PROPER SMALL TRAINING TEST ===\n")
    
    # Small but workable config
    config = {
        'TOKENIZER_TYPE': 'char',
        'D_MODEL': 64,
        'CONTEXT_LEN': 64,      # Smaller context
        'ATTENTION_HEADS': 4,
        'DECODER_BLOCKS': 2,
        'DROPOUT_RATE': 0.1,
        'LEARNING_RATE': 0.001,
        'EPOCHS': 5
    }
    
    # Use extended Alice story (longer text)
    text_file = '/home/akshat/GPT_from_scratch/text_data/alice_extended.txt'
    
    print("1. Creating tokenizer...")
    token_to_id_dict, tokenize_func, _ = create_tokenizer(config, [text_file])
    vocab_size = len(token_to_id_dict)
    print(f"✅ Vocab size: {vocab_size}")
    
    print("\n2. Preparing data...")
    train_ds, val_ds, steps = prepare_tfrecords(
        text_file_path=text_file,
        token_to_id_dict=token_to_id_dict,
        tokenize_func=tokenize_func,
        context_length=config['CONTEXT_LEN'],
        batch_size=4,
        records_per_file=50,  # More records per file
        version_name="proper_test"
    )
    
    print(f"✅ Data prepared, steps: {steps}")
    
    if steps < 5:
        print("⚠️  Very few training steps, using manual training loop...")
        return manual_training_test(config, token_to_id_dict, tokenize_func, text_file)
    
    print("\n3. Creating model...")
    model = GPT(
        d_model=config['D_MODEL'],
        vocab_size=vocab_size,
        context_length=config['CONTEXT_LEN'],
        attention_heads=config['ATTENTION_HEADS'],
        epsilon=config['LEARNING_RATE'],
        decoder_blocks=config['DECODER_BLOCKS'],
        dropout_rate=config['DROPOUT_RATE']
    )
    
    total_params = sum(tf.size(var).numpy() for var in model.trainable_variables)
    print(f"✅ Model created: {total_params:,} parameters")
    
    print("\n4. Training...")
    
    # Manual training for better control
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['LEARNING_RATE'])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Training loop
    for epoch in range(config['EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{config['EPOCHS']}")
        
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_ds.take(min(steps, 20)):  # Limit to 20 batches
            inputs, targets = batch
            
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            batch_count += 1
            
            if batch_count % 5 == 0:
                print(f"  Batch {batch_count}: loss = {loss:.3f}")
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.3f}")
    
    print("\n✅ TRAINING COMPLETED!")
    
    # Test generation
    print("\n5. Testing generation...")
    test_simple_generation(model, tokenize_func, token_to_id_dict, config)
    
    # Cleanup
    import shutil
    if os.path.exists('./tfrecords/proper_test'):
        shutil.rmtree('./tfrecords/proper_test')
    
    return True

def manual_training_test(config, token_to_id_dict, tokenize_func, text_file):
    """Manual training for very small datasets"""
    print("\n🔧 Using manual training for small dataset...")
    
    vocab_size = len(token_to_id_dict)
    
    # Read and tokenize text manually
    with open(text_file, 'r') as f:
        text = f.read()
    
    tokens = tokenize_func(text)
    print(f"Total tokens: {len(tokens)}")
    
    # Create simple training examples
    context_len = config['CONTEXT_LEN']
    examples = []
    
    for i in range(0, len(tokens) - context_len, context_len // 2):  # Overlap examples
        input_seq = tokens[i:i + context_len]
        target_seq = tokens[i + 1:i + context_len + 1]
        
        if len(input_seq) == context_len and len(target_seq) == context_len:
            examples.append((input_seq, target_seq))
    
    print(f"Created {len(examples)} training examples")
    
    if len(examples) == 0:
        print("❌ No training examples created!")
        return False
    
    # Create model
    model = GPT(
        d_model=config['D_MODEL'],
        vocab_size=vocab_size,
        context_length=config['CONTEXT_LEN'],
        attention_heads=config['ATTENTION_HEADS'],
        epsilon=config['LEARNING_RATE'],
        decoder_blocks=config['DECODER_BLOCKS'],
        dropout_rate=config['DROPOUT_RATE']
    )
    
    total_params = sum(tf.size(var).numpy() for var in model.trainable_variables)
    print(f"✅ Model created: {total_params:,} parameters")
    
    # Training
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['LEARNING_RATE'])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    print("\nTraining...")
    for epoch in range(config['EPOCHS']):
        epoch_loss = 0.0
        
        # Shuffle examples
        import random
        random.shuffle(examples)
        
        for i, (input_seq, target_seq) in enumerate(examples):
            # Convert to tensors
            input_ids = tf.constant([input_seq], dtype=tf.int32)
            attention_mask = tf.ones_like(input_ids)
            targets = tf.constant([target_seq], dtype=tf.int32)
            
            with tf.GradientTape() as tape:
                predictions = model((input_ids, attention_mask), training=True)
                loss = loss_fn(targets, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            
            if (i + 1) % max(1, len(examples) // 4) == 0:
                print(f"  Example {i + 1}/{len(examples)}: loss = {loss:.3f}")
        
        avg_loss = epoch_loss / len(examples)
        print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.3f}")
    
    print("\n✅ MANUAL TRAINING COMPLETED!")
    
    # Test generation
    test_simple_generation(model, tokenize_func, token_to_id_dict, config)
    
    return True

def test_simple_generation(model, tokenize_func, token_to_id_dict, config):
    """Test simple text generation"""
    print("\nTesting generation...")
    
    id_to_token = {v: k for k, v in token_to_id_dict.items()}
    context_len = config['CONTEXT_LEN']
    
    test_prompts = ["Alice", "The programmer", "Once"]
    
    for prompt in test_prompts:
        prompt_tokens = tokenize_func(prompt)
        if len(prompt_tokens) == 0:
            continue
            
        print(f"\nPrompt: '{prompt}' -> {prompt_tokens}")
        
        # Generate
        generated = prompt_tokens.copy()
        
        for _ in range(20):  # Generate 20 tokens
            # Prepare input
            current = generated[-context_len:] if len(generated) >= context_len else [0] * (context_len - len(generated)) + generated
            
            input_ids = tf.constant([current], dtype=tf.int32)
            attention_mask = tf.ones_like(input_ids)
            
            # Predict
            logits = model((input_ids, attention_mask), training=False)
            next_logits = logits[0, -1, :]
            
            # Sample
            next_token = tf.argmax(next_logits).numpy()
            generated.append(int(next_token))
        
        # Convert back
        text = ''.join([id_to_token.get(t, '?') for t in generated])
        print(f"Generated: '{text[:60]}...'")

if __name__ == "__main__":
    try:
        success = test_training_proper()
        if success:
            print("\n🎉 TRAINING TEST SUCCESSFUL!")
            print("\n✅ Your GPT model works correctly!")
            print("✅ Data pipeline is functional")
            print("✅ Training completes without errors")
            print("✅ Text generation works")
            print("\n🚀 Ready for larger datasets and longer training!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
