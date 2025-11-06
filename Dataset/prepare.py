import tiktoken
import numpy as np
import os

# 1. Initialize the tokenizer
# 'o200k_base' is the tokenizer used by modern models like GPT-3.5 and GPT-4
enc = tiktoken.get_encoding("o200k_base")
print(f"Tokenizer loaded. Vocabulary size: {enc.n_vocab}")

# 2. Load your dataset
with open('Dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Loaded dataset with {len(text)} characters.")

# 3. Tokenize the entire dataset
tokens = enc.encode(text,allowed_special={'<|endoftext|>'})
print(f"Tokenized into {len(tokens)} tokens.")

# 4. Convert to numpy array for saving
tokens = np.array(tokens, dtype=np.uint32)

# 5. Split into train and validation sets
train_split = int(0.9 * len(tokens))
train_data = tokens[:train_split]
val_data = tokens[train_split:]
print(f"Training set has {len(train_data)} tokens.")
print(f"Validation set has {len(val_data)} tokens.")

# 6. Save to binary files
train_data.tofile('train.bin')
val_data.tofile('val.bin')
print(" train.bin and val.bin created successfully.")

# --- Quick Validation ---
# Let's see how it tokenizes a piece of Python code
code_snippet = "def hello_world():"
encoded = enc.encode(code_snippet)
decoded = enc.decode(encoded)

print("\n--- Tokenizer Test ---")
print(f"Original: {code_snippet}")
print(f"Encoded tokens: {encoded}")
print(f"Decoded: {decoded}")