import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tiktoken

# --- 1. Configuration ---
torch.manual_seed(1337)
batch_size = 24  
block_size = 256     
n_embd = 384

learning_rate = 3e-4
n_heads = 6
n_layers = 6
dropout = 0.2



max_iters = 5000         
eval_interval = 500
eval_iters = 200      
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
if torch.backends.mps.is_available(): # Check for Apple Silicon GPU
    device = 'mps'
print(f"Using device: {device}")

# We need the vocab size to build the model
enc = tiktoken.get_encoding("o200k_base")
model_vocab_size = enc.n_vocab
print(f"Tokenizer vocab size: {model_vocab_size}")

# --- 2. Data Loading ---
# Load data from .bin files
train_data_np = np.fromfile('train.bin', dtype=np.uint32)
val_data_np = np.fromfile('val.bin', dtype=np.uint32)

# Convert to PyTorch tensors
train_data = torch.from_numpy(train_data_np.astype(np.int64))
val_data = torch.from_numpy(val_data_np.astype(np.int64))
print("Data loaded into tensors.")

# --- 3. The `get_batch` function ---
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Move batches to the correct device
    x, y = x.to(device), y.to(device)
    return x, y


# --- Attention Head ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, T)

        # Apply the causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # since this is here this is a decoder only model

        # Softmax to get probabilities
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, head_size)

        # Weighted sum of values
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module): #multiple parallel communication layer
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module): #computation layer
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #expand to 4 times the embedding dimension of the inner layer
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module): #communication followed by computation
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- simple Language Model ---
# This is our version of BigramLanguageModel, adapted for a large vocab
class SimpleLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()


        self.token_embedding_table = nn.Embedding(model_vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        # 2. Language Model Head: (EmbDim, VocabSize)
        self.lm_head = nn.Linear(n_embd, model_vocab_size)


    def forward(self, idx, targets=None):
        # idx is (B, T) tensor of token IDs
        B, T = idx.shape
        # 1. Get Token Embeddings
        # tok_emb shape: (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x= tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x)
        
        # 2. Get Logits
        # We pass the embeddings through the LM head
        # logits shape: (B, T, vocab_size)
        logits = self.lm_head(x)

        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # Calculate the loss
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # --- Get the predictions (forward pass) ---
            idx_cond = idx[:, -block_size:] # crop to the last block_size tokens
            logits, loss = self(idx_cond)
            
            # --- Focus only on the last time step ---
            # The model predicts the next token for *every* token in the sequence
            # We only care about the *last* one.
            logits_last_step = logits[:, -1, :] # becomes (B, C)
            
            # --- Apply softmax to get probabilities ---
            probs = F.softmax(logits_last_step, dim=-1) # (B, C)
            
            # --- Sample from the distribution ---
            # This is where we "roll the dice"
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # --- Append sampled index to the running sequence ---
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

# Create the model and move it to the device
m = SimpleLanguageModel()
m = m.to(device)

# Get a batch of data
xb, yb = get_batch('train')

# Pass the batch through the model
logits, loss = m(xb, yb)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


@torch.no_grad()  # Tell PyTorch we don't need to calculate gradients here
def estimate_loss():
    out = {}
    m.eval()  # Set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() # Set the model back to training mode
    return out

# --- 8. The Training Loop ---
print("Starting training loop...")
for iter in range(max_iters):

    # Every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 1. Get a batch of data
    xb, yb = get_batch('train')

    # 2. Forward pass: evaluate the loss
    logits, loss = m(xb, yb)

    # 3. Zero gradients from previous step
    optimizer.zero_grad(set_to_none=True)

    # 4. Backward pass: get gradients
    loss.backward()

    # 5. Update weights
    optimizer.step()

print("Training finished.")


enc = tiktoken.get_encoding("o200k_base")

# Start with a single "newline" token
# We use device=device to make sure it's on the GPU/MPS
start_context = torch.tensor(enc.encode('\n'), dtype=torch.long, device=device).unsqueeze(0)

# Generate 100 new tokens
generated_tokens_tensor = m.generate(idx=start_context, max_new_tokens=100)

# The output is (B, T), so we get the first (and only) batch
generated_tokens_list = generated_tokens_tensor[0].tolist()

# Decode the tokens back into text
generated_text = enc.decode(generated_tokens_list)
print(generated_text)
