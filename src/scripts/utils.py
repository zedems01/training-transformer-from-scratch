import torch
from pathlib import Path
import logging


def create_pad_mask(seq, pad_idx=1):
    """
    Create a padding mask for attention: (B, 1, 1, T)
    True/1 = valid token, False/0 = padded token
    Compatible with attention scores broadcasting
    """
    mask = (seq != pad_idx)     # (B, T)
    return mask.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, T)

def create_subsequent_mask(seq_len, device):
    """
    Create causal mask for decoder's self-attention: (1, 1, T, T)
    True/1 = allowed position, False/0 = masked future position
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) == 0
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


# def find_latest_checkpoints(checkpoints_dir):
#     """Find the latest checkpoints in the checkpoints directory."""
#     checkpoints_dir = Path(checkpoints_dir)
#     if not checkpoints_dir.exists():
#         raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
#     checkpoints = list(checkpoints_dir.glob("*.pt"))
#     if not checkpoints:
#         raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    
#     latest_checkpoints = max(checkpoints, key=lambda p: p.stat().st_mtime)
#     return latest_checkpoints

# def load_checkpoints(checkpoints_path, model, device):
#     """Load model checkpoints."""
#     checkpoints_path = Path(checkpoints_path)
#     if not checkpoints_path.exists():
#         raise FileNotFoundError(f"Checkpoints not found: {checkpoints_path}")
    
#     logging.info(f"Loading checkpoints from: {checkpoints_path}")
#     checkpoints = torch.load(checkpoints_path, map_location=device)
#     model.load_state_dict(checkpoints['model_state_dict'])
    
#     epoch = checkpoints['epoch']
#     loss = checkpoints['loss']
#     config = checkpoints['model_config']
#     logging.info(f"Checkpoints loaded - Epoch: {epoch}, Loss: {loss}")
    
#     return model, config


def greedy_decode(model, src, src_mask, max_len, device, bos_id=2, eos_id=3):
    """
    Greedy decoding for translation 
    
    Args:
        model: Transformer model
        src: Source tensor (B, T_src)
        src_mask: Source mask (B, 1, 1, T_src)
        max_len: Maximum length for generation
        device: Device to run on
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
    
    Returns:
        Decoded sequences (B, T_tgt)
    """
    model.eval()
    batch_size = src.size(0)
    
    with torch.no_grad():
        enc_output = model.encoder(src, mask=src_mask)
    
    # start with BOS token for each sequence in batch
    tgt = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    
    # track which sequences have finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len - 1):
        tgt_mask = create_subsequent_mask(tgt.size(1), device)
        
        # Decode - get only last token logits to save memory
        with torch.no_grad():
            output = model.decoder(tgt, enc_output, tgt_mask=tgt_mask, enc_mask=src_mask)
            last_token_logits = output[:, -1, :]  # (B, vocab_size)
            del output
        next_token = last_token_logits.argmax(dim=-1)  # (B,)
        finished = finished | (next_token == eos_id)
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        if finished.all():
            break
    
    return tgt