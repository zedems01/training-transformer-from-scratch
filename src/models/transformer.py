import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        # shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
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

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (B, T, d_model)
        mask: (B, ..., T_q, T_k), broadcasted automatically
        """
        B, T_q, _ = q.size()
        _, T_k, _ = k.size()
        
        # Linear projections
        q = self.query(q).view(B, T_q, self.num_heads, self.head_dim)
        k = self.key(k).view(B, T_k, self.num_heads, self.head_dim)
        v = self.value(v).view(B, T_k, self.num_heads, self.head_dim)

        # Reshape for multi-head: (B, T, num_heads, head_dim) --> (B, num_heads, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        ) # Shape: (B, num_heads, T_q, head_dim)

        # Recombine the heads
        # (B, num_heads, T_q, head_dim) -> (B, T_q, num_heads, head_dim) -> (B, T_q, d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T_q, self.d_model)

        # Final linear projection
        out = self.out(attn_output)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        # Feed Forward
        ff_out = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, enc_mask=None):
        # Self-attention
        _x = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(_x + self.dropout(x))
        # Cross-attention
        _x = x
        x = self.cross_attn(x, enc_output, enc_output, enc_mask)
        x = self.norm2(_x + self.dropout(x))
        # Feed Forward
        _x = x
        x = self.ffn(x)
        x = self.norm3(_x + self.dropout(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T) token indices
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, tgt_mask=None, enc_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, enc_mask)
        logits = self.fc_out(x)  # (B, T, vocab_size)
        return logits


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src_mask: 4D encoder mask of shape (B, 1, T_src, T_src),
        tgt_mask: 4D decoder mask of shape (B, 1, T_tgt, T_tgt).
        """
        enc_output = self.encoder(src, mask=src_mask)
        out = self.decoder(tgt, enc_output, tgt_mask=tgt_mask)
        return out
    
