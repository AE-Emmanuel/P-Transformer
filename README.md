# P-Transformer: A GPT-style Model for Python Code

This project is a complete implementation of a decoder-only transformer (like GPT) built from scratch in PyTorch. It follows the educational structure of Andrej Karpathy's "Let's build GPT" video, but is adapted to train on a large-scale dataset of Python code instead of Shakespeare.

The final model is a 6-layer, 6-head transformer with ~77 million parameters, trained on a subset of "The Stack" dataset.

## 🚀 Final Result

After training for 5,000 steps on a single NVIDIA A100 GPU, the model achieved:
* **Final Validation Loss:** `4.5826`

It is capable of generating coherent, syntactically-aware (though not perfect) Python code snippets:

```python
        try:
            logging.info(" 0-13: Train value returned.")
            raise ValueError(
                "Checkpoint probs selected errors "
                "Setting value setting type you Born arguments)
            )
            try:
                log.warning(
                    "The specified exception and will not current node, node option `%n".join(
                                       loc, wait_for_tub.id))

    except AttributeError("error cannot be requested after but if one of "
                    "ellipsisapy generic ensure that contains classes.")

🏗️ Architecture
This model is a decoder-only transformer built from the ground up. All key components are implemented from scratch in train.py:

Tokenization: tiktoken (o200k_base)

Embeddings: Token and Positional Embeddings

Self-Attention Head: Scaled Dot-Product Attention

Multi-Head Attention: Parallel attention heads with projection

Feed-Forward Network: Standard 4x "thinking" layer

Transformer Block: The complete unit with residual connections and layer normalization

Training: Full training and validation loops with the AdamW optimizer.

💻 How to Run
This project is designed to be reproduced. The data files (.bin, .txt) are not included in the repository and must be generated.

1. Clone the Repository:

Bash

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
2. Install Dependencies: This project uses torch, numpy, and tiktoken.

Bash

pip install torch numpy tiktoken
3. Generate the Dataset: This process is in three steps and requires you to have a Hugging Face account (for init_dataset.py). Note: The data files are ignored by .gitignore.

Bash

# 1. Download ~1M lines of Python code from Hugging Face
# This creates input.txt (or use your own)
# (You'll need to provide your own download script, e.g., init_dataset.py)
python Dataset/init_dataset.py 

# 2. Tokenize the data
# This reads input.txt and creates train.bin and val.bin
python Dataset/prepare.py
4. Start Training: It is highly recommended to run this on a CUDA-enabled GPU (e.g., in Google Colab).

Bash

python train.py
💡 Future Work
The v1.0 model is a strong baseline, but it's significantly undertrained. The next steps to improve performance would be:

Train Longer: 5,000 iterations is just a start. Training for 50,000 or 100,000 steps would yield a much lower loss.

Implement Learning Rate Scheduler: Add a cosine decay scheduler to the training loop to fine-tune the model as loss plateaus.

Increase Data: Train on a much larger subset of "The Stack" (e.g., 10GB of Python code instead of ~1M lines).