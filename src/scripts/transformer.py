import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        # shape(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, offset: int=0) -> torch.Tensor:
        """
        Args:
            x shape: (batch_size, seq_len, d_model), (B, T, d_model)
            offset: positional offset to apply during inference with KV cache
        """
        seq_len = x.size(1)
        x = x + self.pe[:, offset:offset + seq_len, :]
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor]=None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            q, k, v: (B, T, d_model)
            mask: (B, ..., T_q, T_k), broadcasted automatically
            past_kv: tuple of (cached_k, cached_v) or None
        
        Returns:
            A tuple containing:
            - The attention output tensor of shape (B, T_q, d_model).
            - The present Key and Value tensors for caching, of shape
              (B, num_heads, T_new, head_dim).
        """
        B, T_q, _ = q.size()
        _, T_k, _ = k.size()

        # linear projections
        q = self.query(q).view(B, T_q, self.num_heads, self.head_dim)
        k = self.key(k).view(B, T_k, self.num_heads, self.head_dim)
        v = self.value(v).view(B, T_k, self.num_heads, self.head_dim)

        # reshape for multi-head
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, T_q, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, num_heads, T_k, head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, num_heads, T_v, head_dim)

        # handle KV cache for efficient inference
        if past_kv is not None:
            past_k, past_v = past_kv
            # concatenate past and present key/value states
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # full concatenated key/value state
        present_kv = (k, v)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T_q, self.d_model)
        out = self.out(attn_output)

        return out, present_kv


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # self-attention without cache
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        # feed forward
        ff_out = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor]=None,
                enc_mask: Optional[torch.Tensor]=None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            A tuple containing:
            - The decoder layer output tensor.
            - The present self-attention Key/Value state for caching.
        """
        # self-attention with kv cache
        _x = x
        x, present_kv = self.self_attn(q=x, k=x, v=x, mask=tgt_mask, past_kv=past_kv)
        x = self.norm1(_x + self.dropout(x))
        
        # cross-attention without recursive cache
        # K/V always come from `enc_output`
        _x = x
        x, _ = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=enc_mask)
        x = self.norm2(_x + self.dropout(x))
        
        # feed forward
        _x = x
        x = self.ffn(x)
        x = self.norm3(_x + self.dropout(x))
        
        return x, present_kv


class TransformerEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float
            ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # x: (B, T) token indices
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                vocab_size: int,
                d_model: int,
                num_heads: int,
                d_ff: int,
                num_layers: int,
                dropout: float
            ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self,
                x: torch.Tensor,
                enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                enc_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
               ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            A tuple containing:
            - The output logits tensor.
            - The list of present Key/Value states for all decoder layers.
        """
        # determine offset for positional encoding based on cache length
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x, offset=past_len)
        x = self.dropout(x)

        present_key_values = []

        for i, layer in enumerate(self.layers):
            # get the cache for the current layer
            past_kv = past_key_values[i] if past_key_values is not None else None
            # forward pass through the layer, getting output and new cache
            x, present_kv = layer(x, enc_output, tgt_mask, enc_mask, past_kv=past_kv)
            present_key_values.append(present_kv)

        logits = self.fc_out(x)
        return logits, present_key_values


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float
                 ):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor]=None,
                tgt_mask: Optional[torch.Tensor]=None
            ) -> torch.Tensor:
        """
        This forward method is primarily for training and does not use the KV cache.
        For inference, the encoder and decoder will be called separately.
        src_mask: 4D encoder mask of shape (B, 1, T_src, T_src),
        tgt_mask: 4D decoder mask of shape (B, 1, T_tgt, T_tgt).
        """
        enc_output = self.encoder(src, mask=src_mask)
        logits, _ = self.decoder(tgt, enc_output, tgt_mask=tgt_mask)
        return logits
    

