import os
import numpy as np
import tensorflow as tf

def build_vocab(text_path):
    with open(text_path, encoding='utf-8') as f:
        text = f.read()
    vocab = sorted(set(text))
    token_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_token = {i: ch for ch, i in token_to_id.items()}
    print(f'Vocab size: {len(vocab)}')
    return token_to_id, id_to_token, vocab

def tokenize_and_pad(text_batch, token_to_id, max_seq_len, pad_value=0):
    batch_token_ids = []
    for text in text_batch:
        ids = [token_to_id.get(c, pad_value) for c in text]
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        else:
            ids += [pad_value] * (max_seq_len - len(ids))
        batch_token_ids.append(ids)
    token_ids = np.array(batch_token_ids, dtype=np.int32)
    attention_mask = (token_ids != pad_value).astype(np.int32)
    print('Token IDs:', token_ids)
    print('Attention Mask:', attention_mask)
    return token_ids, attention_mask

def prepare_training_data(text, token_to_id, context_length=10, pad_value=0):
    token_ids = [token_to_id.get(c, pad_value) for c in text]
    inputs, targets = [], []
    for i in range(0, len(token_ids) - context_length):
        input_seq = token_ids[i:i+context_length]
        target_seq = token_ids[i+1:i+context_length+1]
        inputs.append(input_seq)
        targets.append(target_seq)
    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    print('Inputs:', inputs)
    print('Targets:', targets)
    return inputs, targets
