import torch


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


def greedy_decode(model, src, src_mask, max_len, device, bos_id, eos_id):
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
    
    # list of tuples (K, V) for each decoder layer
    past_key_values = None
    # complete sequence of generated tokens
    generated_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    # input to the decoder for the next step, len 1
    next_input = generated_tokens
    # track which sequences have finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len - 1):
        with torch.no_grad():
            # only pass the last token and the cache to get the logits and updated cache
            logits, past_key_values = model.decoder(
                x=next_input, 
                enc_output=enc_output,
                enc_mask=src_mask,
                past_key_values=past_key_values
            )
            last_token_logits = logits[:, -1, :]  # (B, vocab_size)

        next_token = last_token_logits.argmax(dim=-1)  # (B,)
        finished = finished | (next_token == eos_id)
        
        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
        next_input = next_token.unsqueeze(1)
        
        if finished.all():
            break
            
    return generated_tokens




    #         output = model.decoder(tgt, enc_output, tgt_mask=tgt_mask, enc_mask=src_mask)
    #         last_token_logits = output[:, -1, :]  # (B, vocab_size)
    #         del output
    #     next_token = last_token_logits.argmax(dim=-1)  # (B,)
    #     finished = finished | (next_token == eos_id)
    #     tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

    #     if finished.all():
    #         break
    
    # return tgt