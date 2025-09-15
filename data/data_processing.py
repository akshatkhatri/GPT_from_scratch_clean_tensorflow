import tensorflow as tf
import numpy as np
from typing import Dict, List
import os
from tqdm import tqdm


def create_tf_example(input_ids: List[int], target_ids: List[int], attention_mask: List[int]) -> bytes:
    feature = {
        'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
        'target_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=target_ids)),
        'attention_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask))
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def convert_text_to_tfrecord(
    text_file_path: str,
    token_to_id_dict: Dict[str, int],
    tokenize_func,
    output_dir: str,
    context_length: int = 512,
    records_per_file: int = 1000,
    pad_value: int = 0
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading text file: {text_file_path}")
    
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text length: {len(text):,} characters")
    
    print("Tokenizing text...")
    token_ids = tokenize_func(text)
    print(f"Token length: {len(token_ids):,} tokens")
    
    num_examples = (len(token_ids) - context_length) // context_length
    print(f"Will create {num_examples:,} training examples")
    
    file_count = 0
    examples_in_current_file = 0
    writer = None
    
    print("Creating TFRecord files...")
    
    for i in tqdm(range(0, len(token_ids) - context_length, context_length)):
        
        window = token_ids[i:i + context_length + 1]
        if len(window) < context_length + 1:
            break
            
        input_ids = window[:-1]
        target_ids = window[1:]
        attention_mask = [1] * context_length
        
        if writer is None or examples_in_current_file >= records_per_file:
            if writer is not None:
                writer.close()
            
            tfrecord_filename = os.path.join(output_dir, f'train_{file_count:04d}.tfrecord')
            writer = tf.io.TFRecordWriter(tfrecord_filename)
            file_count += 1
            examples_in_current_file = 0
        
        # Write training example to current TFRecord file
        tf_example = create_tf_example(input_ids, target_ids, attention_mask)
        writer.write(tf_example) # type: ignore
        examples_in_current_file += 1
    
    # Cleanup
    if writer is not None:
        writer.close()
    
    # Step 4: Save summary information
    print(f"\nConversion complete!")
    print(f"Created {file_count} TFRecord files in: {output_dir}")
    print(f"Total examples: {num_examples:,}")
    
    # Write metadata file
    metadata = {
        'context_length': context_length,
        'vocab_size': len(token_to_id_dict),
        'num_examples': num_examples,
        'num_files': file_count,
        'records_per_file': records_per_file
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Metadata saved to: {metadata_path}")
    
    return output_dir


def create_tf_data_pipeline(
    tfrecord_dir: str,
    context_length: int = 512,
    batch_size: int = 32,
    shuffle_buffer: int = 1000,
    prefetch_buffer: int = tf.data.AUTOTUNE
) -> tf.data.Dataset:
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    print(f"Found {len(tfrecord_files)} TFRecord files")
    feature_description = {
        'input_ids': tf.io.FixedLenFeature([context_length], tf.int64),
        'target_ids': tf.io.FixedLenFeature([context_length], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([context_length], tf.int64)
    }
    
    def parse_tfrecord_example(example_proto):
        """Parse a single TFRecord example into model inputs and targets."""
        # Parse the serialized example
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Convert to correct data types
        input_ids = tf.cast(parsed_features['input_ids'], tf.int32)
        target_ids = tf.cast(parsed_features['target_ids'], tf.int32)
        attention_mask = tf.cast(parsed_features['attention_mask'], tf.int32)
        
        # Return in format expected by your GPT model: ((input_ids, attention_mask), targets)
        model_inputs = (input_ids, attention_mask)
        model_targets = target_ids
        
        return model_inputs, model_targets
    
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    
    return dataset

def create_train_val_datasets(tfrecord_dir: str, 
                             context_length: int,
                             batch_size: int = 32,
                             val_split: float = 0.1):
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    tfrecord_files = sorted(tfrecord_files)
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    num_val_files = max(1, int(len(tfrecord_files) * val_split))
    val_files = tfrecord_files[:num_val_files]
    train_files = tfrecord_files[num_val_files:]
    
    print(f"Using {len(train_files)} files for training, {len(val_files)} for validation")
    
    feature_description = {
        'input_ids': tf.io.FixedLenFeature([context_length], tf.int64),
        'target_ids': tf.io.FixedLenFeature([context_length], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([context_length], tf.int64)
    }
    
    def parse_function(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        input_ids = tf.cast(parsed_features['input_ids'], tf.int32)
        target_ids = tf.cast(parsed_features['target_ids'], tf.int32)
        attention_mask = tf.cast(parsed_features['attention_mask'], tf.int32)
        
        return (input_ids, attention_mask), target_ids
    
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(5000)
    train_dataset = train_dataset.repeat()  # Repeat to prevent running out of data
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_dataset = tf.data.TFRecordDataset(val_files)
    val_dataset = val_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.repeat()  # Repeat to prevent running out of data
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    metadata_path = os.path.join(tfrecord_dir, 'metadata.txt')
    train_example_count = 0
    
    if os.path.exists(metadata_path):
        metadata = {}
        with open(metadata_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    try:
                        metadata[key.strip()] = int(value.strip())
                    except ValueError:
                        metadata[key.strip()] = value.strip()
        
        total_examples = metadata.get('num_examples', 0)
        train_example_count = int(total_examples * len(train_files) / len(tfrecord_files))
        print(f"Using metadata for counting: {train_example_count} estimated training examples")
    else:
        print("Metadata not found, counting examples...")
        for tfrecord_file in train_files:
            for _ in tf.data.TFRecordDataset(tfrecord_file):
                train_example_count += 1
        print(f"Counted {train_example_count} training examples")
    
    steps_per_epoch = max(1, train_example_count // batch_size)
    
    if steps_per_epoch < 5:
        print(f"⚠️  WARNING: Very few steps per epoch ({steps_per_epoch}). Consider:")
        print(f"   - Reducing batch_size (current: {batch_size})")
        print(f"   - Reducing context_length (current: {context_length})")
        print(f"   - Using more training data")
    
    print(f"Training examples: {train_example_count}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    return train_dataset, val_dataset, steps_per_epoch

import os
import tensorflow as tf

def prepare_tfrecords(
    text_file_path: str,
    token_to_id_dict: dict,
    tokenize_func,
    context_length: int = 128,
    records_per_file: int = 1000,
    output_base_dir: str = './tfrecords',
    version_name: str | None = None,
    batch_size: int = 16,
    val_split: float = 0.1
):
    if version_name is None:
        version_name = f"context_{context_length}_bs{batch_size}"
    output_dir = os.path.join(output_base_dir, version_name)
    os.makedirs(output_dir, exist_ok=True)

    tfrecord_dir = convert_text_to_tfrecord(
        text_file_path=text_file_path,
        token_to_id_dict=token_to_id_dict,
        tokenize_func=tokenize_func,
        output_dir=output_dir,
        context_length=context_length,
        records_per_file=records_per_file
    )

    train_dataset, val_dataset, steps_per_epoch = create_train_val_datasets(
        tfrecord_dir=tfrecord_dir,
        context_length=context_length,
        batch_size=batch_size,
        val_split=val_split
    )
    print(f"TFRecord folder: {tfrecord_dir}")

    return train_dataset, val_dataset, steps_per_epoch

