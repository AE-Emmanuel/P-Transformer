import tiktoken
import numpy as np
import os

# Initialize the tokenizer
enc = tiktoken.get_encoding("o200k_base")
print(f"Tokenizer loaded. Vocabulary size: {enc.n_vocab}")

# Load your dataset
with open('Dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Loaded dataset with {len(text)} characters.")

# Tokenize the entire dataset
tokens = enc.encode(text,allowed_special={'<|endoftext|>'})
print(f"Tokenized into {len(tokens)} tokens.")

# Convert to numpy array for saving
tokens = np.array(tokens, dtype=np.uint32)

# Split into train and validation sets
train_split = int(0.9 * len(tokens))
train_data = tokens[:train_split]
val_data = tokens[train_split:]
print(f"Training set has {len(train_data)} tokens.")
print(f"Validation set has {len(val_data)} tokens.")

# Save to binary files
train_data.tofile('train.bin')
val_data.tofile('val.bin')

