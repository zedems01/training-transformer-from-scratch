import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from tqdm import tqdm
from pathlib import Path
from src.models.transformer import Transformer
from src.config import PROCESSED_DATA_DIR, CHECKPOINTS_DIR

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# special token IDs
UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3

class TranslationDataset(Dataset):
    """
    Dataset for BPE-encoded parallel translation data
    Uses the SentencePiece model trained in preprocess_data.py
    """
    def __init__(self, src_file, tgt_file, sp_model_path, max_len):
        super().__init__()
        self.max_len = max_len
        self.sp_model_path = sp_model_path
        
        # load SentencePiece model and BPE-encoded data
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.src_data = []
        self.tgt_data = []
        
        filtered_count = 0
        with Path(src_file).open('r', encoding='utf-8') as src_f, \
             Path(tgt_file).open('r', encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                # BPE tokens are already space-separated in the .bpe files
                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()
                # filter by max length (accounting for BOS/EOS tokens)
                if len(src_tokens) <= max_len - 2 and len(tgt_tokens) <= max_len - 2:
                    # convert BPE tokens to indices
                    src_indices = [self.sp.piece_to_id(token) for token in src_tokens]
                    tgt_indices = [self.sp.piece_to_id(token) for token in tgt_tokens]
                    self.src_data.append(src_indices)
                    self.tgt_data.append(tgt_indices)
                else:
                    filtered_count += 1
        
        logging.info(f"Loaded {len(self.src_data)} sentence pairs\n")
        logging.info(f"Filtered out {filtered_count} sequences exceeding max_len={max_len}\n")

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # add BOS and EOS tokens to target sequence
        src_indices = self.src_data[idx]
        tgt_indices = [BOS_ID] + self.tgt_data[idx] + [EOS_ID]
        return src_indices, tgt_indices

    @staticmethod
    def collate_fn(batch):
        """Collate batch with padding"""
        src_batch, tgt_batch = zip(*batch)
        # convert to tensors
        src_tensors = [torch.tensor(seq, dtype=torch.long) for seq in src_batch]
        tgt_tensors = [torch.tensor(seq, dtype=torch.long) for seq in tgt_batch]
        # batch_first=True for (B, T) shape
        src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_ID)
        tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=PAD_ID)

        return src_padded, tgt_padded


def create_pad_mask(seq, pad_idx=PAD_ID):
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


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    for batch_idx, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(device), tgt.to(device)
        
        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        # source padding mask: (B, 1, 1, T_src)
        src_mask = create_pad_mask(src, pad_idx=PAD_ID)
        # target padding mask: (B, 1, 1, T_tgt)
        tgt_pad_mask = create_pad_mask(tgt_input, pad_idx=PAD_ID)
        # causal mask: (1, 1, T_tgt, T_tgt)
        T_tgt = tgt_input.size(1)
        causal_mask = create_subsequent_mask(T_tgt, device)
        tgt_mask = tgt_pad_mask & causal_mask

        # forward pass
        optimizer.zero_grad()
        outputs = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        # reshape outputs and labels for loss computation
        outputs = outputs.reshape(-1, outputs.size(-1))
        tgt_labels = tgt_labels.reshape(-1)
        # compute loss and backpropagate
        loss = criterion(outputs, tgt_labels)
        loss.backward()
        # gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # update model parameters
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Validating", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask = create_pad_mask(src, pad_idx=PAD_ID)
            tgt_pad_mask = create_pad_mask(tgt_input, pad_idx=PAD_ID)
            T_tgt = tgt_input.size(1)
            causal_mask = create_subsequent_mask(T_tgt, device)
            tgt_mask = tgt_pad_mask & causal_mask

            # forward pass
            outputs = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_labels = tgt_labels.reshape(-1)
            loss = criterion(outputs, tgt_labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = checkpoint_dir / f'transformer_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f"Checkpoint loaded from epoch {epoch}, loss: {loss:.3f}")
    return epoch, loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                         else "cpu")
    logging.info(f"Using device: {device}")

    # Hyperparameters
    SP_MODEL_PATH = PROCESSED_DATA_DIR / "sp_bpe.model"
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    DROPOUT = 0.1
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
    MAX_SEQ_LEN = 512

    if not SP_MODEL_PATH.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {SP_MODEL_PATH}. "
                               f"Run preprocess_data.py first.")

    # load SentencePiece model to get vocab size
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL_PATH)
    actual_vocab_size = sp.get_piece_size()
    logging.info(f"SentencePiece model loaded with vocabulary size: {actual_vocab_size}")

    train_dataset = TranslationDataset(
        src_file=PROCESSED_DATA_DIR / "train.en.bpe",
        tgt_file=PROCESSED_DATA_DIR / "train.fr.bpe",
        sp_model_path=SP_MODEL_PATH,
        max_len=MAX_SEQ_LEN
    )
    
    val_dataset = TranslationDataset(
        src_file=PROCESSED_DATA_DIR / "valid.en.bpe",
        tgt_file=PROCESSED_DATA_DIR / "valid.fr.bpe",
        sp_model_path=SP_MODEL_PATH,
        max_len=MAX_SEQ_LEN
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0
    )
    
    model = Transformer(
        src_vocab_size=actual_vocab_size,
        tgt_vocab_size=actual_vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # initialize optimizer with warmup
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    # use label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    # training loop
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"{'='*50}")
        logging.info(f"Epoch {epoch}/{EPOCHS}")
        logging.info(f"{'='*50}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        logging.info(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir=CHECKPOINTS_DIR)
            logging.info(f"New best validation loss: {val_loss:.3f}")

    logging.info("Training complete!")


if __name__ == "__main__":
    main()