# P-Transformer: A GPT for Python

This project is a complete implementation of a decoder-only transformer built from scratch in PyTorch. It follows instructions of Andrej Karpathy's "Let's build GPT" video, but is adapted to train on a large-scale dataset -> bigcode/the-stack-v2-dedup and also has some other major changes with tokenization.

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
## Architecture (brief)
This model is a decoder-only transformer built from the ground up. All key components are implemented from scratch in train.py:

Tokenization: tiktoken (o200k_base)

Embeddings: Token and Positional Embeddings

Self-Attention Head: Scaled Dot-Product Attention

Multi-Head Attention: Parallel attention heads with projection

### Note:
This v1.0 of the model cause it's significantly undertrained.
I created this model to learn the core architecture by experimenting it with some new datasets 
