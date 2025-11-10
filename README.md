# Transformer from scratch

**Open-source reimplementation** of the Transformer architecture, based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), enriched with my personal notes and explanations.


It aims to be:

- **Minimal**: The implementation mirrors the paper's architecture (multi-head attention, positional encodings, and the encoder-decoder design), as a way for me to deepen my own understanding, rather than to build a production-optimized library.
- **Customizable**: Hyperparameters such as the number of layers, hidden size, and dropout can be adjusted to reproduce the base, big, or smaller Transformer variants.


## Requirements

```bash
python -m venv .venv
.venv\Scripts\activate
uv sync
```


## Citation

```bibtex
@article{vaswani2017attention,
title={Attention is all you need},
author={Vaswani, Ashish and others},
journal={Advances in neural information processing systems},
volume={30},
year={2017}
} 
```
