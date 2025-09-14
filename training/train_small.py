#!/usr/bin/env python3
"""
Small-scale training script using Alice story for quick validation
"""

import sys
import os
import tensorflow as tf
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT, CosineDecayWithWarmup

def train_small_model():
    """Train on Alice story with small model for testing"""
    print("=== SMALL-SCALE TRAINING TEST ===\n")
    
    # Load small training config
    config = load_config('/home/akshat/GPT_from_scratch/cleaned_code/configs/config_small_train.txt')
    print(f"Config loaded: {config['TOKENIZER_TYPE']} tokenizer")
    print(f"Model size: {config['D_MODEL']} dim, {config['DECODER_BLOCKS']} blocks")
    
    # Use Alice story
    text_file = '/home/akshat/GPT_from_scratch/text_data/alice_story.txt'
    
    # Create tokenizer
    print("\n1. Creating tokenizer...")
    token_to_id_dict, tokenize_func, extra_info = create_tokenizer(config, [text_file])
    vocab_size = len(token_to_id_dict)
    print(f"✅ Tokenizer created, vocab size: {vocab_size}")
    
    # Prepare data
    print("\n2. Preparing data...")
    train_ds, val_ds, calculated_steps = prepare_tfrecords(
        text_file_path=text_file,
        token_to_id_dict=token_to_id_dict,
        tokenize_func=tokenize_func,
        context_length=config['CONTEXT_LEN'],
        batch_size=4,  # Small batch size
        records_per_file=20,
        version_name="alice_small_train"
    )
    
    # Use actual calculated steps, but limit to config max
    steps_per_epoch = min(config['STEPS_PER_EPOCH'], calculated_steps)
    print(f"✅ Data prepared")
    print(f"   Calculated steps: {calculated_steps}")
    print(f"   Using steps: {steps_per_epoch}")
    
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
    
    total_params = sum(tf.size(var).numpy() for var in model.trainable_variables)
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Create learning rate schedule
    print("\n4. Setting up training...")
    total_steps = steps_per_epoch * config['EPOCHS']
    warmup_steps = int(total_steps * config['WARMUP_RATIO'])
    
    lr_schedule = CosineDecayWithWarmup(
        min_learning_rate=config['MIN_LEARNING_RATE'],
        peak_learning_rate=config['PEAK_LEARNING_RATE'],
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    print(f"✅ Model compiled with learning rate schedule")
    print(f"   Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Setup callbacks
    checkpoint_dir = "./alice_small_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{checkpoint_dir}/model_epoch_{{epoch:02d}}.keras",
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            f"{checkpoint_dir}/training_log.csv",
            append=False
        )
    ]
    
    print("\n5. Starting training...")
    print(f"Training for {config['EPOCHS']} epochs")
    print("-" * 50)
    
    try:
        history = model.fit(
            train_ds.take(steps_per_epoch),
            epochs=config['EPOCHS'],
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds.take(max(1, steps_per_epoch // 4)),
            validation_steps=max(1, steps_per_epoch // 4),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        
        # Show training progress
        print("\nTraining Summary:")
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        print(f"Final loss: {final_loss:.4f}")
        print(f"Final accuracy: {final_acc:.4f}")
        
        # Test generation
        print("\n6. Testing text generation...")
        test_generation(model, tokenize_func, token_to_id_dict, config)
        
        return True, model, history
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_generation(model, tokenize_func, token_to_id_dict, config):
    """Test text generation with the trained model"""
    
    id_to_token = {v: k for k, v in token_to_id_dict.items()}
    context_len = config['CONTEXT_LEN']
    
    test_prompts = [
        "Alice",
        "Once upon a time",
        "The programmer"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize prompt
        prompt_tokens = tokenize_func(prompt)
        if len(prompt_tokens) == 0:
            prompt_tokens = [0]
        
        # Generate text
        generated_tokens = prompt_tokens.copy()
        
        for i in range(30):  # Generate 30 more tokens
            # Prepare input
            current_tokens = generated_tokens[-context_len:]
            if len(current_tokens) < context_len:
                # Pad at the beginning
                padded = [0] * (context_len - len(current_tokens)) + current_tokens
            else:
                padded = current_tokens
            
            input_ids = tf.constant([padded], dtype=tf.int32)
            attention_mask = tf.ones_like(input_ids)
            
            # Predict
            logits = model((input_ids, attention_mask), training=False)
            next_token_logits = logits[0, -1, :]
            
            # Sample with temperature
            temperature = 0.8
            next_token_logits = next_token_logits / temperature
            probs = tf.nn.softmax(next_token_logits)
            
            # Sample from distribution
            next_token = tf.random.categorical([next_token_logits], 1)[0, 0].numpy()
            
            generated_tokens.append(int(next_token))
        
        # Convert to text
        try:
            generated_text = ''.join([id_to_token.get(token_id, '?') for token_id in generated_tokens])
            print(f"Generated: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
        except Exception as e:
            print(f"Generation error: {e}")

def cleanup():
    """Clean up test files"""
    import shutil
    test_dirs = ['./tfrecords/alice_small_train', './alice_small_checkpoints']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"🧹 Cleaned up {test_dir}")
            except Exception as e:
                print(f"Warning: Could not clean {test_dir}: {e}")

if __name__ == "__main__":
    try:
        success, model, history = train_small_model()
        
        if success:
            print("\n🎉 SMALL TRAINING TEST SUCCESSFUL!")
            print("\nKey findings:")
            print("✅ Data pipeline works correctly")
            print("✅ Model architecture is functional")
            print("✅ Training loop executes without errors")
            print("✅ Text generation works")
            print("\n🚀 Ready to scale up to larger datasets!")
        else:
            print("\n💥 Training test failed - check logs above")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    finally:
        print("\nCleaning up...")
        cleanup()
