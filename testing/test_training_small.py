#!/usr/bin/env python3
"""
Test training script for small text file
"""

import sys
import os
import tensorflow as tf
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT

def create_learning_rate_schedule(steps_per_epoch, config):
    """Create simple learning rate schedule"""
    return config['LEARNING_RATE']  # Use constant learning rate for simplicity

def test_training():
    """Test training on a small text file"""
    print("=== SMALL TEXT TRAINING TEST ===\n")
    
    # Load test config
    config = load_config('/home/akshat/GPT_from_scratch/cleaned_code/config_test.txt')
    print(f"✅ Config loaded: {config}")
    
    # Small text file
    text_file = '/home/akshat/GPT_from_scratch/text_data/alice_story.txt'
    
    # Create tokenizer
    print("\n1. Creating tokenizer...")
    token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [text_file])
    vocab_size = len(token_to_id_dict)
    print(f"✅ Tokenizer created, vocab size: {vocab_size}")
    
    # Test tokenization
    with open(text_file, 'r') as f:
        sample_text = f.read()[:200]
    sample_tokens = tokenize_func(sample_text)
    print(f"✅ Sample tokenization: {len(sample_tokens)} tokens from {len(sample_text)} chars")
    
    # Prepare data
    print("\n2. Preparing TFRecords...")
    train_ds, val_ds, steps_per_epoch = prepare_tfrecords(
        text_file_path=text_file,
        token_to_id_dict=token_to_id_dict,
        tokenize_func=tokenize_func,
        context_length=config['CONTEXT_LEN'],
        batch_size=4,  # Small batch for testing
        records_per_file=20,
        version_name="alice_test"
    )
    
    print(f"✅ Data prepared, steps per epoch: {steps_per_epoch}")
    
    # Override steps from config with actual calculated steps
    actual_steps = min(config['STEPS_PER_EPOCH'], steps_per_epoch)
    print(f"Using {actual_steps} steps per epoch (config: {config['STEPS_PER_EPOCH']}, calculated: {steps_per_epoch})")
    
    # Create model
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
    
    print(f"✅ Model created with {sum(tf.keras.utils.count_params(w) for w in model.trainable_weights):,} parameters")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    for batch in train_ds.take(1):
        inputs, targets = batch
        outputs = model(inputs, training=False)
        print(f"✅ Forward pass successful: {outputs.shape}")
        break
    
    # Create learning rate schedule
    print("\n5. Setting up training...")
    lr_schedule = create_learning_rate_schedule(
        steps_per_epoch=actual_steps,
        config=config
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("✅ Model compiled")
    
    # Create callbacks
    checkpoint_path = f"./alice_test_checkpoints/alice_model_epoch_{{epoch:02d}}.keras"
    os.makedirs("./alice_test_checkpoints", exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Start training
    print("\n6. Starting training...")
    print(f"Training for {config['EPOCHS']} epochs with {actual_steps} steps per epoch")
    
    try:
        history = model.fit(
            train_ds.take(actual_steps),
            epochs=config['EPOCHS'],
            steps_per_epoch=actual_steps,
            validation_data=val_ds.take(max(1, actual_steps // 4)),
            validation_steps=max(1, actual_steps // 4),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        
        # Test generation
        print("\n7. Testing text generation...")
        test_generation(model, tokenize_func, token_to_id_dict)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation(model, tokenize_func, token_to_id_dict):
    """Test text generation with the trained model"""
    
    # Create reverse mapping
    id_to_token_dict = {v: k for k, v in token_to_id_dict.items()}
    
    # Test prompts
    test_prompts = [
        "Once upon a time",
        "Alice was",
        "The model"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize prompt
        prompt_tokens = tokenize_func(prompt)
        if len(prompt_tokens) == 0:
            prompt_tokens = [0]  # Fallback
        
        # Pad or truncate to context length
        context_len = 128
        if len(prompt_tokens) > context_len:
            prompt_tokens = prompt_tokens[:context_len]
        else:
            prompt_tokens = prompt_tokens + [0] * (context_len - len(prompt_tokens))
        
        # Generate
        input_ids = tf.constant([prompt_tokens], dtype=tf.int32)
        attention_mask = tf.ones_like(input_ids, dtype=tf.int32)
        
        # Generate a few tokens
        generated_tokens = prompt_tokens[:len(tokenize_func(prompt))]
        
        for _ in range(20):  # Generate 20 more tokens
            # Prepare input
            current_tokens = generated_tokens[-context_len:]
            if len(current_tokens) < context_len:
                current_tokens = [0] * (context_len - len(current_tokens)) + current_tokens
            
            input_ids = tf.constant([current_tokens], dtype=tf.int32)
            attention_mask = tf.ones_like(input_ids, dtype=tf.int32)
            
            # Predict next token
            logits = model((input_ids, attention_mask), training=False)
            next_token_logits = logits[0, -1, :]
            
            # Simple sampling (get most probable token)
            next_token_id = tf.argmax(next_token_logits).numpy()
            
            # Add to generated sequence
            generated_tokens.append(int(next_token_id))
        
        # Convert back to text
        try:
            generated_text = ''.join([id_to_token_dict.get(token_id, '?') for token_id in generated_tokens])
            print(f"Generated: '{generated_text[:100]}...'")
        except Exception as e:
            print(f"Generation error: {e}")

def cleanup():
    """Clean up test files"""
    import shutil
    test_dirs = ['./tfrecords/alice_test', './alice_test_checkpoints']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"🧹 Cleaned up {test_dir}")
            except:
                pass

if __name__ == "__main__":
    try:
        success = test_training()
        if success:
            print("\n🎉 SMALL TEXT TRAINING TEST PASSED!")
            print("Your model is ready for larger datasets!")
        else:
            print("\n💥 TRAINING TEST FAILED!")
    finally:
        cleanup()
