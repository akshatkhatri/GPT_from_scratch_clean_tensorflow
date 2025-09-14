# Plot training curves
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_processing import prepare_tfrecords
from models.utils import build_id_to_token, create_tokenizer, load_config
import tensorflow as tf
import keras
import numpy as np
from models.layers import GPT, CosineDecayWithWarmup

# Load configuration
config = load_config('/home/akshat/GPT_from_scratch/cleaned_code/configs/config.txt')

# Extract model parameters
D_MODEL = config['D_MODEL']
CONTEXT_LEN = config['CONTEXT_LEN']
ATTENTION_HEADS = config['ATTENTION_HEADS']
DECODER_BLOCKS = config['DECODER_BLOCKS']
DROPOUT_RATE = config['DROPOUT_RATE']
LEARNING_RATE = config['LEARNING_RATE']

# Extract training parameters
EPOCHS = config['EPOCHS']
STEPS_PER_EPOCH = config['STEPS_PER_EPOCH']
WARMUP_RATIO = config['WARMUP_RATIO']
PEAK_LEARNING_RATE = config['PEAK_LEARNING_RATE']
MIN_LEARNING_RATE = config['MIN_LEARNING_RATE']

# Extract tokenizer configuration
TOKENIZER_TYPE = config['TOKENIZER_TYPE']
VOCAB_SIZE_CONFIG = config.get('VOCAB_SIZE', 2000)  # Only used for word/sentencepiece

FILE_PATH = '/home/akshat/GPT_from_scratch/text_data/jane_austen_clean.txt'

# Create tokenizer based on configuration
print(f"Creating {TOKENIZER_TYPE} tokenizer...")
token_to_id_dict, tokenize_func, tokenizer_extra = create_tokenizer(config, [FILE_PATH])

# Calculate vocab size from the actual vocabulary
VOCAB_SIZE = len(token_to_id_dict)
print(f"Vocabulary size: {VOCAB_SIZE}")

# Print tokenizer info
if TOKENIZER_TYPE != 'char':
    print(f"Configured vocab size: {VOCAB_SIZE_CONFIG}")
    if VOCAB_SIZE != VOCAB_SIZE_CONFIG and TOKENIZER_TYPE != 'char':
        print(f"Note: Actual vocab size ({VOCAB_SIZE}) may differ from configured ({VOCAB_SIZE_CONFIG})")

train_ds_64, val_ds_64, steps_64 = prepare_tfrecords(
    text_file_path=FILE_PATH,
    token_to_id_dict=token_to_id_dict,
    tokenize_func=tokenize_func,
    context_length=CONTEXT_LEN,
    batch_size=64,
)

# Now calculate learning rate schedule with actual steps
TOTAL_STEPS = EPOCHS * steps_64
WARMUP_STEPS = int(WARMUP_RATIO * TOTAL_STEPS)

# Create the learning rate schedule
lr_schedule = CosineDecayWithWarmup(
    warmup_steps=WARMUP_STEPS,
    total_steps=TOTAL_STEPS,
    peak_learning_rate=PEAK_LEARNING_RATE,
    min_learning_rate=MIN_LEARNING_RATE
)

train_ds_64 = train_ds_64.shuffle(10000)
val_ds_64 = val_ds_64.shuffle(10000)

model = GPT(D_MODEL, VOCAB_SIZE, CONTEXT_LEN, ATTENTION_HEADS, LEARNING_RATE, DECODER_BLOCKS, DROPOUT_RATE)

# Compile your model (same as before)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0) # type: ignore
model.compile(optimizer=opt, loss=loss,metrics=['accuracy']) # type: ignore

callbacks = [
    # Save model every epoch with epoch number
    keras.callbacks.ModelCheckpoint(
        filepath='/home/akshat/GPT_from_scratch/notebooks/rewrite_char_level_checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras',
        save_freq='epoch',
        save_best_only=False,
        verbose=1
    ),
    
    # Save best model separately
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # More reasonable early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # Wait 50 epochs before stopping
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.CSVLogger('training_log.csv'),
    keras.callbacks.TensorBoard(
        log_dir="./char_logs", 
        histogram_freq=1, 
        profile_batch=0,
        write_graph=True
    )
]

# Training
print("Starting training...")
print(f"Total epochs: {EPOCHS}")
print(f"Steps per epoch: {steps_64}")
print(f"Total steps: {EPOCHS * steps_64}")

history = model.fit(
    train_ds_64,
    validation_data=val_ds_64,
    epochs=EPOCHS,
    steps_per_epoch=steps_64,  # Use calculated steps, not config value
    callbacks=callbacks,
    verbose=1 # type: ignore
)

print("Training completed!")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
if 'learning_rate' in history.history:
    plt.plot(history.history['learning_rate'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

plt.subplot(1, 3, 3)
train_perplexity = [np.exp(loss) for loss in history.history['loss']]
val_perplexity = [np.exp(loss) for loss in history.history['val_loss']]
plt.plot(train_perplexity, label='Train Perplexity')
plt.plot(val_perplexity, label='Val Perplexity')
plt.title('Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.tight_layout()
plt.show()