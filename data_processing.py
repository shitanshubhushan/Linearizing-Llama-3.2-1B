# data_processing.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataset import LLMDataset

def collect_tokens(tokenizer, target_tokens=5_000):
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",
                          name="sample-10BT",
                          split="train",
                          streaming=True)
    
    collected_tokens = []
    total_tokens = 0
    
    for sample in dataset:
        text = sample['text']
        tokens = tokenizer(text, truncation=False, padding=False)['input_ids']
        collected_tokens.extend(tokens)
        total_tokens += len(tokens)
        
        if total_tokens // 100_000 > (total_tokens - len(tokens)) // 100_000:
            print(f"Collected {total_tokens:,} tokens")
            
        if total_tokens >= target_tokens:
            break
            
    return np.array(collected_tokens[:target_tokens])

def create_sequences(collected_tokens, sequence_length=512, stride=128):
    n_sequences = (len(collected_tokens) - sequence_length) // stride + 1
    sequences = []
    
    for i in range(0, len(collected_tokens) - sequence_length + 1, stride):
        sequence = collected_tokens[i:i + sequence_length]
        sequences.append(sequence)
        
    sequences = np.array(sequences)
    input_sequences = sequences[:, :-1]
    target_sequences = sequences[:, 1:]
    
    inputs = torch.tensor(input_sequences)
    masks = torch.tensor(np.ones_like(input_sequences))
    targets = torch.tensor(target_sequences)
    
    return inputs, masks, targets

def create_data_loaders(inputs, masks, targets, batch_size=2, val_split=0.1):
    val_idx = int(len(inputs) * (1 - val_split))
    
    # Split into train/val
    train_inputs = inputs[:val_idx]
    train_masks = masks[:val_idx]
    train_targets = targets[:val_idx]
    val_inputs = inputs[val_idx:]
    val_masks = masks[val_idx:]
    val_targets = targets[val_idx:]
    
    # Create datasets
    train_dataset = LLMDataset(train_inputs, train_masks, train_targets)
    val_dataset = LLMDataset(val_inputs, val_masks, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader