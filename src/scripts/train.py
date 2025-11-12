import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import sentencepiece as spm

from src.scripts.utils import create_pad_mask, create_subsequent_mask
from src.scripts.transformer import Transformer
from src.config import (
    CHECKPOINTS_DIR,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    D_MODEL,
    NUM_HEADS,
    D_FF,
    NUM_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    LR,
    MAX_SEQ_LEN,
    WARMUP_STEPS,
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    SP_MODEL_PATH,
    TRAIN_SRC_FILE,
    TRAIN_TGT_FILE,
    VALID_SRC_FILE,
    VALID_TGT_FILE,
)


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
        self.sp.load(str(sp_model_path))
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


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        """
        Call this after each validation
        Returns True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.verbose:
                logging.info(f"Initial validation loss: {val_loss:.3f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info(f"Early stopping triggered! Best epoch: {self.best_epoch}")
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logging.info(f"Validation loss improved to {val_loss:.3f}")
        
        return self.early_stop


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
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
        # update model parameters and learning rate
        optimizer.step()
        scheduler.step()

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


def save_checkpoints(model, optimizer, epoch, loss, config, checkpoints_dir, model_name=None):
    """Save model checkpoint."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': config
    }
    if not model_name:
        model_name = 'best'
    checkpoints_path = checkpoints_dir / f'{model_name}.pt'
    torch.save(checkpoints, checkpoints_path)
    logging.info(f"Checkpoint saved: {checkpoints_path}")


def train_transformer(config):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.device("cuda")
    logging.info(f"Using device: {device}")

    d_model = config['d_model']
    num_heads = config['heads']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    dropout = config['dropout']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    max_seq_len = config['max_seq_len']
    warmup_steps = config['warmup_steps']
    early_stopping_patience = config['early_stopping_patience']
    num_workers = config['num_workers']
    sp_model_path = Path(config['sp_model_path'])
    train_src_file = Path(config['train_src_file'])
    train_tgt_file = Path(config['train_tgt_file'])
    valid_src_file = Path(config['valid_src_file'])
    valid_tgt_file = Path(config['valid_tgt_file'])
    checkpoints_dir = Path(config['checkpoints_dir'])
    model_name = config['model_name']

    if not sp_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {sp_model_path}. "
                               f"Run preprocess_data.py first.")

    # load SentencePiece model to get vocab size
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    actual_vocab_size = sp.get_piece_size()
    logging.info(f"SentencePiece model loaded with vocabulary size: {actual_vocab_size}")

    train_dataset = TranslationDataset(
        src_file=train_src_file,
        tgt_file=train_tgt_file,
        sp_model_path=sp_model_path,
        max_len=max_seq_len
    )

    val_dataset = TranslationDataset(
        src_file=valid_src_file,
        tgt_file=valid_tgt_file,
        sp_model_path=sp_model_path,
        max_len=max_seq_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TranslationDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    model = Transformer(
        src_vocab_size=actual_vocab_size,
        tgt_vocab_size=actual_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f} M)")
    logging.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.1f} M)")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    # create learning rate scheduler with linear warmup
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    logging.info(f"Scheduler created: {warmup_steps} warmup steps, {total_steps} total steps")
    
    # use label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    # training loop
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        logging.info(f"{'='*50}")
        logging.info(f"Epoch {epoch}/{epochs}")
        logging.info(f"{'='*50}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        logging.info(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        
        # save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoints(model, optimizer, epoch, val_loss, config, checkpoints_dir, model_name)
        
        if early_stopping(val_loss, epoch):
            logging.info(f"Early stopping at epoch {epoch}")
            break

    logging.info(f"Training complete! Best validation loss: {best_val_loss:.3f}")

    
def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer for English-French translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--d-model', type=int, default=D_MODEL,
                        help='Dimension of model embeddings and layers')
    parser.add_argument('--heads', type=int, default=NUM_HEADS,
                        help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=D_FF,
                        help='Dimension of feedforward network')
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS,
                        help='Number of encoder/decoder layers')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Learning rate')
    parser.add_argument('--max-seq-len', type=int, default=MAX_SEQ_LEN,
                        help='Maximum sequence length')
    parser.add_argument('--warmup-steps', type=int, default=WARMUP_STEPS,
                        help='Number of warmup steps')
    parser.add_argument('--early-stopping-patience', type=int, default=EARLY_STOPPING_PATIENCE,
                        help='Number of epochs to wait for early stopping')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help='Number of workers for data loading')
    parser.add_argument('--model-name', type=str, default='best',
                        help='Name of the model, not including the extension')
    
    args = parser.parse_args()
    config = args.__dict__
    config['sp_model_path'] = SP_MODEL_PATH
    config['checkpoints_dir'] = CHECKPOINTS_DIR
    config['train_src_file'] = TRAIN_SRC_FILE
    config['train_tgt_file'] = TRAIN_TGT_FILE
    config['valid_src_file'] = VALID_SRC_FILE
    config['valid_tgt_file'] = VALID_TGT_FILE
    
    train_transformer(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    main()