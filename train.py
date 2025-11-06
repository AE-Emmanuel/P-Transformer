import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tiktoken

# --- Configuration ---
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
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
if torch.backends.mps.is_available(): 
    device = 'mps'
print(f"Using device: {device}")

# We need the vocab size to build the model
enc = tiktoken.get_encoding("o200k_base")
model_vocab_size = enc.n_vocab
print(f"Tokenizer vocab size: {model_vocab_size}")

# --- Data Loading ---
# Load data from .bin files
train_data_np = np.fromfile('train.bin', dtype=np.uint32)
val_data_np = np.fromfile('val.bin', dtype=np.uint32)

# Convert to PyTorch tensors
train_data = torch.from_numpy(train_data_np.astype(np.int64))
val_data = torch.from_numpy(val_data_np.astype(np.int64))
print("Data loaded into tensors.")

# --- The `get_batch` function ---
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
        k = self.key(x)   
        q = self.query(x) 

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 

        # Apply the causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # since this is here this is a decoder only model

        # Softmax to get probabilities
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)

        v = self.value(x) 

        # Weighted sum of values
        out = wei @ v 
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
            nn.Linear(n_embd, 4 * n_embd), 
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
        # Language Model Head: (EmbDim, VocabSize)
        self.lm_head = nn.Linear(n_embd, model_vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape
        # Get Token Embeddings
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) 
        x= tok_emb + pos_emb
        x = self.blocks(x) 
        x = self.ln_f(x)
        
        # Get Logits
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
        
        for _ in range(max_new_tokens):
            # --- Get the predictions (forward pass) ---
            idx_cond = idx[:, -block_size:] 
            logits, loss = self(idx_cond)
            logits_last_step = logits[:, -1, :]
            
            # --- Apply softmax to get probabilities ---
            probs = F.softmax(logits_last_step, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # --- Append sampled index to the running sequence ---
            idx = torch.cat((idx, idx_next), dim=1) 
            
        return idx

# Create the model and move it to the device
m = SimpleLanguageModel()
m = m.to(device)

# Get a batch of data
xb, yb = get_batch('train')

# Pass the batch through the model
logits, loss = m(xb, yb)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


@torch.no_grad()  
def estimate_loss():
    out = {}
    m.eval()  
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() 
    return out

# ---The Training Loop ---

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get a batch of data
    xb, yb = get_batch('train')

    # Forward pass: evaluate the loss
    logits, loss = m(xb, yb)

    #  Zero gradients from previous step
    optimizer.zero_grad(set_to_none=True)

    #  Backward pass: get gradients
    loss.backward()

    #  Update weights
    optimizer.step()

print("Training finished.")


enc = tiktoken.get_encoding("o200k_base")

# Start with a single "newline" token
start_context = torch.tensor(enc.encode('\n'), dtype=torch.long, device=device).unsqueeze(0)

# Generate 100 new tokens
generated_tokens_tensor = m.generate(idx=start_context, max_new_tokens=100)

# The output is (B, T), so we get the first (and only) batch
generated_tokens_list = generated_tokens_tensor[0].tolist()

# Decode the tokens back into text
generated_text = enc.decode(generated_tokens_list)
print(generated_text)
