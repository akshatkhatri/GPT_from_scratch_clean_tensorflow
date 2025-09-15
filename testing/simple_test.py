#!/usr/bin/env python3
"""
Simple training test for small text - very lightweight
"""

import sys
import os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT

def simple_test():
    """Simple test with very small model"""
    print("=== SIMPLE SMALL MODEL TEST ===\n")
    
    # Use Alice story
    text_file = '/home/akshat/GPT_from_scratch/text_data/alice_story.txt'
    
    # Minimal config for testing
    config = {
        'TOKENIZER_TYPE': 'char',
        'VOCAB_SIZE': 1000,
        'D_MODEL': 32,        # Very small
        'CONTEXT_LEN': 64,    # Very small
        'ATTENTION_HEADS': 2, # Very small
        'DECODER_BLOCKS': 1,  # Very small
        'DROPOUT_RATE': 0.1,
        'LEARNING_RATE': 0.001,
        'EPOCHS': 3           # Just 3 epochs
    }
    
    print(f"✅ Using minimal config: {config}")
    
    print("\n1. Creating tokenizer...")
    token_to_id_dict, tokenize_func, _ = create_tokenizer(config, [text_file])
    vocab_size = len(token_to_id_dict)
    print(f"✅ Vocab size: {vocab_size}")
    
    with open(text_file, 'r') as f:
        text = f.read()
    print(f"✅ Text length: {len(text)} characters")
    
    print("\n2. Preparing data...")
    train_ds, val_ds, steps = prepare_tfrecords(
        text_file_path=text_file,
        token_to_id_dict=token_to_id_dict,
        tokenize_func=tokenize_func,
        context_length=config['CONTEXT_LEN'],
        batch_size=2,  # Very small batch
        records_per_file=10,
        version_name="simple_test"
    )
    
    print(f"✅ Data ready, steps: {steps}")
    actual_steps = min(10, steps)  # Max 10 steps for test
    print(f"Using {actual_steps} steps for training")
    
    # Create tiny model
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
    
    # Count parameters manually to avoid keras import issues
    total_params = sum(tf.size(var).numpy() for var in model.trainable_variables)
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    for batch in train_ds.take(1):
        inputs, targets = batch
        print(f"Input shape: {inputs[0].shape}")
        outputs = model(inputs, training=False)
        print(f"✅ Forward pass OK: {outputs.shape}")
        
        # Test loss calculation
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = loss_fn(targets, outputs)
        print(f"✅ Loss: {loss:.3f}")
        break
    
    # Simple training
    print("\n5. Simple training test...")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['LEARNING_RATE'])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Manual training loop (more memory efficient)
    print("Starting manual training loop...")
    
    for epoch in range(config['EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{config['EPOCHS']}")
        
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_ds.take(actual_steps):
            inputs, targets = batch
            
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            batch_count += 1
            
            if batch_count % 5 == 0 or batch_count == actual_steps:
                print(f"  Batch {batch_count}: loss = {loss:.3f}")
        
        avg_loss = epoch_loss / batch_count
        print(f"  Epoch {epoch + 1} average loss: {avg_loss:.3f}")
    
    print("\n✅ TRAINING COMPLETED!")
    
    # Test generation
    print("\n6. Testing generation...")
    test_prompt = "Alice"
    prompt_tokens = tokenize_func(test_prompt)
    print(f"Prompt '{test_prompt}' -> {prompt_tokens}")
    
    # Simple generation
    if len(prompt_tokens) > 0:
        # Pad prompt
        context_len = config['CONTEXT_LEN']
        if len(prompt_tokens) < context_len:
            padded_tokens = [0] * (context_len - len(prompt_tokens)) + prompt_tokens
        else:
            padded_tokens = prompt_tokens[:context_len]
        
        input_ids = tf.constant([padded_tokens], dtype=tf.int32)
        attention_mask = tf.ones_like(input_ids)
        
        # Generate one token
        logits = model((input_ids, attention_mask), training=False)
        next_token_logits = logits[0, -1, :]
        next_token = tf.argmax(next_token_logits).numpy()
        
        print(f"✅ Generated next token: {next_token}")
        
        # Convert back to char
        id_to_token = {v: k for k, v in token_to_id_dict.items()}
        if next_token in id_to_token:
            next_char = id_to_token[next_token]
            print(f"✅ Next character: '{next_char}'")
    
    # Cleanup
    import shutil
    if os.path.exists('./tfrecords/simple_test'):
        shutil.rmtree('./tfrecords/simple_test')
    
    print("\n🎉 SIMPLE TEST COMPLETED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    try:
        success = simple_test()
        if success:
            print("\n✅ Your model architecture works!")
            print("Ready for larger datasets and longer training!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
