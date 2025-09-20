import os
import numpy as np
import tensorflow as tf
from collections import Counter

def build_word_vocab(text_path, max_words=10000):
    with open(text_path, encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(max_words)
    vocab = [w for w, _ in most_common]
    token_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_token = {i: word for word, i in token_to_id.items()}
    print(f'Vocabulary size (limited): {len(vocab)}')
    return token_to_id, id_to_token, vocab

def tokenize_and_pad_words(text_batch, token_to_id, max_seq_len, pad_value=0):
    batch_token_ids = []
    for text in text_batch:
        words = text.split()
        ids = [token_to_id.get(w, pad_value) for w in words]
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        else:
            ids += [pad_value] * (max_seq_len - len(ids))
        batch_token_ids.append(ids)
    token_ids = np.array(batch_token_ids, dtype=np.int32)
    attention_mask = (token_ids != pad_value).astype(np.int32)
    print('Word Token IDs:', token_ids)
    print('Word Attention Mask:', attention_mask)
    return token_ids, attention_mask

def prepare_word_training_data(text, token_to_id, context_length=6, pad_value=0):
    words = text.split()
    word_ids = [token_to_id.get(w, pad_value) for w in words]
    inputs, targets = [], []
    for i in range(0, len(word_ids) - context_length):
        input_seq = word_ids[i:i+context_length]
        target_seq = word_ids[i+1:i+context_length+1]
        print(f'Window {i}:')
        print(f'  Input IDs: {input_seq}')
        print(f'  Target IDs: {target_seq}')
        inputs.append(input_seq)
        targets.append(target_seq)
    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    print('Final input array:', inputs)
    print('Final target array:', targets)
    return inputs, targets
