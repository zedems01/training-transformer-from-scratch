import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from tqdm import tqdm
import logging
from src.models.transformer import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# Special token IDs from preprocess_data.py
UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3


class TranslationDataset(Dataset):
    """
    Dataset for BPE-encoded parallel translation data.
    Uses the SentencePiece model trained in preprocess_data.py.
    """
    def __init__(self, src_file, tgt_file, sp_model_path, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.sp_model_path = sp_model_path
        
        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        
        # Load BPE-encoded data
        self.src_data = []
        self.tgt_data = []
        
        filtered_count = 0
        with open(src_file, 'r', encoding='utf-8') as src_f, \
             open(tgt_file, 'r', encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                # BPE tokens are already space-separated in the .bpe files
                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()
                
                # Filter by max length (accounting for BOS/EOS tokens)
                if len(src_tokens) <= max_len - 2 and len(tgt_tokens) <= max_len - 2:
                    # Convert BPE tokens to indices
                    src_indices = [self.sp.piece_to_id(token) for token in src_tokens]
                    tgt_indices = [self.sp.piece_to_id(token) for token in tgt_tokens]
                    
                    self.src_data.append(src_indices)
                    self.tgt_data.append(tgt_indices)
                else:
                    filtered_count += 1
        
        logging.info(f"Loaded {len(self.src_data)} sentence pairs")
        logging.info(f"Filtered out {filtered_count} sequences exceeding max_len={max_len}")

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # Add BOS and EOS tokens to target sequence
        src_indices = self.src_data[idx]
        tgt_indices = [BOS_ID] + self.tgt_data[idx] + [EOS_ID]
        return src_indices, tgt_indices

    @staticmethod
    def collate_fn(batch):
        """Collate batch with padding."""
        src_seqs, tgt_seqs = zip(*batch)
        
        # Get max lengths in this batch
        src_max_len = max(len(seq) for seq in src_seqs)
        tgt_max_len = max(len(seq) for seq in tgt_seqs)

        # Initialize with PAD_ID (1)
        src_padded = torch.full((len(batch), src_max_len), PAD_ID, dtype=torch.long)
        tgt_padded = torch.full((len(batch), tgt_max_len), PAD_ID, dtype=torch.long)

        for i, (src_seq, tgt_seq) in enumerate(zip(src_seqs, tgt_seqs)):
            src_padded[i, :len(src_seq)] = torch.tensor(src_seq, dtype=torch.long)
            tgt_padded[i, :len(tgt_seq)] = torch.tensor(tgt_seq, dtype=torch.long)

        return src_padded, tgt_padded

def create_pad_mask(seq, pad_idx=PAD_ID):
    """
    Creates a 2D pad mask: (B, T), where 1 = valid token, 0 = pad.
    """
    return (seq != pad_idx).long()


def expand_pad_mask(mask_2d):
    """
    Convert a (B, T) 2D mask into (B, 1, T, T) for self-attention.
    1 = allowed token, 0 = masked/padded.
    """
    B, T = mask_2d.shape
    mask_4d = mask_2d.unsqueeze(1).unsqueeze(2).expand(B, 1, T, T)
    return mask_4d


def generate_subsequent_mask(size, device):
    """Generate causal mask for decoder's self attention."""
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask.unsqueeze(0)  # (1, size, size)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    for batch_idx, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(device), tgt.to(device)
        
        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        # Create source padding mask
        src_pad_2d = create_pad_mask(src, pad_idx=PAD_ID)
        src_mask_4d = expand_pad_mask(src_pad_2d).to(device)

        # Create target padding mask
        tgt_pad_2d = create_pad_mask(tgt_input, pad_idx=PAD_ID)
        tgt_pad_4d = expand_pad_mask(tgt_pad_2d)

        # Generate causal mask
        T_tgt = tgt_input.size(1)
        B = tgt_input.size(0)
        causal_3d = generate_subsequent_mask(T_tgt, device)
        causal_4d = causal_3d.expand(B, 1, -1, -1)

        # Combine padding and causal masks
        final_tgt_mask = (tgt_pad_4d * causal_4d).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(src, tgt_input, src_mask=src_mask_4d, tgt_mask=final_tgt_mask)
        
        # Compute loss
        outputs = outputs.reshape(-1, outputs.size(-1))
        tgt_labels = tgt_labels.reshape(-1)
        
        loss = criterion(outputs, tgt_labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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

            # Create masks
            src_pad_2d = create_pad_mask(src, pad_idx=PAD_ID)
            src_mask_4d = expand_pad_mask(src_pad_2d).to(device)

            tgt_pad_2d = create_pad_mask(tgt_input, pad_idx=PAD_ID)
            tgt_pad_4d = expand_pad_mask(tgt_pad_2d)

            T_tgt = tgt_input.size(1)
            B = tgt_input.size(0)
            causal_3d = generate_subsequent_mask(T_tgt, device)
            causal_4d = causal_3d.expand(B, 1, -1, -1)

            final_tgt_mask = (tgt_pad_4d * causal_4d).to(device)

            # Forward pass
            outputs = model(src, tgt_input, src_mask=src_mask_4d, tgt_mask=final_tgt_mask)
            
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_labels = tgt_labels.reshape(-1)
            
            loss = criterion(outputs, tgt_labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="./checkpoints"):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'transformer_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f"Checkpoint loaded from epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    logging.info(f"Using device: {device}")

    # Hyperparameters
    DATA_DIR = "./src/data/processed"
    SP_MODEL_PATH = os.path.join(DATA_DIR, "sp_bpe.model")
    VOCAB_SIZE = 32000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    DROPOUT = 0.1
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    MAX_SEQ_LEN = 128
    WARMUP_STEPS = 4000

    # Verify files exist
    if not os.path.exists(SP_MODEL_PATH):
        raise FileNotFoundError(f"SentencePiece model not found at {SP_MODEL_PATH}. "
                               f"Please run preprocess_data.py first.")

    # Load SentencePiece model to get vocab size
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL_PATH)
    actual_vocab_size = sp.get_piece_size()
    logging.info(f"Loaded SentencePiece model with vocabulary size: {actual_vocab_size}")

    # Create datasets
    train_dataset = TranslationDataset(
        src_file=os.path.join(DATA_DIR, "train.en.bpe"),
        tgt_file=os.path.join(DATA_DIR, "train.fr.bpe"),
        sp_model_path=SP_MODEL_PATH,
        max_len=MAX_SEQ_LEN
    )
    
    val_dataset = TranslationDataset(
        src_file=os.path.join(DATA_DIR, "valid.en.bpe"),
        tgt_file=os.path.join(DATA_DIR, "valid.fr.bpe"),
        sp_model_path=SP_MODEL_PATH,
        max_len=MAX_SEQ_LEN
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=0
    )
    
    # Create model
    model = Transformer(
        src_vocab_size=actual_vocab_size,
        tgt_vocab_size=actual_vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer with warmup
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    
    # Use label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\n{'='*50}")
        logging.info(f"Epoch {epoch}/{EPOCHS}")
        logging.info(f"{'='*50}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir="./checkpoints")
            logging.info(f"New best validation loss: {val_loss:.4f}")

    logging.info("\nTraining complete!")


if __name__ == "__main__":
    main()