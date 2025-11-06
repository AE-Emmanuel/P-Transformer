# P-Transformer: A GPT for Python Code

This project is a complete implementation of a decoder-only transformer built from scratch in PyTorch. It follows instructions of Andrej Karpathy's "Let's build GPT" video, but is adapted to train on a large-scale dataset -> bigcode/the-stack-v2-dedup.

The final model is a 6-layer, 6-head transformer with ~77 million parameters.

## 🚀 Final Result

After training for 5,000 steps on a single A100 GPU, the model achieved:
* **Final Validation Loss:** `4.5826`

It is capable of generating coherent, syntactically-aware Python code snippets :

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
```
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
# P-Transformer

A compact, from-scratch implementation of a decoder-only transformer (GPT-style) written in PyTorch and targeted for training on Python source code.

This repo follows the educational "Let's build GPT" approach while adapting the data pipeline and training loop to work with large-scale Python code datasets (e.g., a subset of The Stack).

Key points:
- Model: Decoder-only transformer (6 layers, 6 heads, ~77M parameters in the reference configuration).
- Tokenizer: tiktoken (o200k_base) used for byte-pair-like tokenization of Python source.
- Purpose: Research / education — a reproducible baseline for code modelling.

## Highlights
- Reimplements core transformer blocks (attention, MLP, residuals, layer norm) in `train.py`.
- Small, reproducible training setup so you can iterate locally or scale to a GPU instance.
- Dataset pipeline under `Dataset/` to collect, tokenize, and produce binary train/validation files.

## Results
In an initial run (5,000 steps on a single A100) the project logged a final validation loss of `4.5826`. This demonstrates a working end-to-end pipeline and a model that generates syntactically-aware Python snippets (still imperfect — more training and data will improve performance).

## Quickstart
Prerequisites:
- Python 3.9+ (or your environment's supported interpreter)
- A CUDA-capable GPU with an appropriate PyTorch build for reasonable training speed (optional but recommended)

Install (example):

```bash
python -m pip install --upgrade pip
pip install torch numpy tiktoken
```

Generate dataset (high-level):

1. Create or download a source file of Python code lines (the repo expects `Dataset/input.txt`).
2. Run the dataset preparation scripts to produce tokenized binaries.

```bash
# download or assemble raw python lines into Dataset/input.txt
python Dataset/init_dataset.py   # optional helper (may require Hugging Face creds)
python Dataset/prepare.py        # tokenizes and writes train.bin, val.bin
```

Train the model:

```bash
python train.py
```

Notes:
- The repo's data files (train.bin, val.bin) are not committed by default; you must generate them.
- For meaningful training, use a GPU and increase the number of steps and dataset size.

## Project layout
- `train.py` — model definition and training loop (core of the project).
- `Dataset/` — dataset helper scripts:
  - `init_dataset.py` — (optional) download/collect raw python lines
  - `prepare.py` — tokenizes `input.txt` into `train.bin` and `val.bin`
  - `verify_data.py` — small utilities to sanity-check the dataset
- `README.md` — this file

## Architecture (brief)
- Token embedding + positional embedding
- Multi-head self-attention (scaled dot-product)
- Feed-forward MLP (typically 4x hidden dim)
- Residual connections and layer normalization
- AdamW optimizer and basic training/validation loops

## Tips to improve results
- Train longer and on more data: current runs are small for quick iteration.
- Use a learning rate schedule (cosine decay / warmup) to stabilize training.
- Increase model capacity (layers/heads/embedding size) if you have more compute.

## Reproducibility & dataset
This repo is intended as a reproducible baseline. The dataset scripts assume you will provide or download Python source lines; see `Dataset/init_dataset.py` for one approach. The binary files created by `prepare.py` (e.g., `train.bin` / `val.bin`) are what `train.py` consumes.

## Contributing
- Open an issue describing the change you'd like to make.
- Small fixes or clarifications to the README are welcome.

## License
The project does not include an explicit license file in the repo snapshot here. If you plan to share this project publicly, add a `LICENSE` file with your chosen license (MIT is a common choice for research code).

## Next steps (suggestions)
1. Add a small `requirements.txt` or pin dependencies in `pyproject.toml` to make reproducible installs easier.
2. Add a short `examples/` notebook to demonstrate dataset preparation and a tiny inference example using a trained checkpoint.
3. Add minimal unit tests for dataset prep and tokenization to catch regressions.

If you want, I can apply these suggestions (create `requirements.txt`, add a short example notebook, or add tests) next.