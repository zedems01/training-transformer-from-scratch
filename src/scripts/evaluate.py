import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import sacrebleu
import sentencepiece as spm

from src.scripts.utils import create_pad_mask, create_subsequent_mask, find_latest_checkpoints, load_checkpoints, greedy_decode

from src.scripts.transformer import Transformer
from src.config import (
    PROCESSED_DATA_DIR,
    CHECKPOINTS_DIR,
    UNK_ID,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    D_MODEL,
    NUM_HEADS,
    D_FF,
    NUM_LAYERS,
    DROPOUT,
    EVAL_BATCH_SIZE,
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
    TEST_SRC_FILE,
    TEST_TGT_FILE,
    TEST_TGT_RAW_FILE,
)



def load_test_data(src_file, tgt_raw_file, sp_model_path, max_len):
    """
    Load and prepare test data.
    
    Returns:
        src_data: List of source token ID sequences
        tgt_texts: List of original target texts (references)
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    
    src_data = []
    tgt_texts = []
    
    with src_file.open('r', encoding='utf-8') as src_f, \
         tgt_raw_file.open('r', encoding='utf-8') as tgt_raw_f:
        
        filtered_count = 0
        for src_line, tgt_raw_line in zip(src_f, tgt_raw_f):
            src_tokens = src_line.strip().split()
            
            if len(src_tokens) <= max_len - 2:
                src_indices = [sp.piece_to_id(token) for token in src_tokens]
                src_data.append(src_indices)
                tgt_texts.append(tgt_raw_line.strip())
            else:
                filtered_count += 1
    
    logging.info(f"Loaded {len(src_data)} test samples")
    logging.info(f"Filtered out {filtered_count} sequences exceeding max_len={max_len}")
    
    return src_data, tgt_texts


def evaluate_on_test_set(
    model,
    sp_model_path,
    test_src_file,
    test_tgt_raw_file,
    device,
    max_seq_len,
    batch_size=16,
    num_samples=5
):
    """
    Evaluate model on test set and compute BLEU score.
    
    Args:
        model: Trained Transformer model
        sp_model_path: Path to SentencePiece model
        test_src_file: Path to test source file
        test_tgt_raw_file: Path to test target raw file
        device: Device to run on
        max_seq_len: Maximum sequence length
        batch_size: Batch size for inference
        num_samples: Number of sample translations to display
    
    Returns:
        BLEU score
    """


    
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    src_data, tgt_texts = load_test_data(
        test_src_file, test_tgt_raw_file, sp_model_path, max_seq_len
    )
    
    # src_data, tgt_texts = load_test_data(
    #     test_src_file, test_tgt_file, 
    #     test_src_file.parent / test_src_file.name.replace('.bpe', ''),
    #     test_tgt_file.parent / test_tgt_file.name.replace('.bpe', ''),
    #     sp_model_path, max_seq_len
    # )
    
    model.eval()
    
    all_hypotheses = []
    all_references = tgt_texts
    
    num_batches = (len(src_data) + batch_size - 1) // batch_size
    
    logging.info("Generating translations...")
    for batch_idx in tqdm(range(num_batches), desc="Translating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(src_data))
        batch_src = src_data[start_idx:end_idx]
        
        batch_src_tensors = [torch.tensor(seq, dtype=torch.long) for seq in batch_src]
        padded_src = pad_sequence(batch_src_tensors, batch_first=True, padding_value=PAD_ID)
        padded_src = padded_src.to(device)
        
        src_mask = create_pad_mask(padded_src, pad_idx=PAD_ID)
        
        output = greedy_decode(
            model, padded_src, src_mask, 
            max_len=max_seq_len, device=device,
            bos_id=BOS_ID, eos_id=EOS_ID
        )
        
        output_cpu = output.cpu().tolist()
        
        # decode outputs
        for i in range(len(batch_src)):
            output_ids = output_cpu[i]
            
            if output_ids[0] == BOS_ID:
                output_ids = output_ids[1:]
            if EOS_ID in output_ids:
                eos_idx = output_ids.index(EOS_ID)
                output_ids = output_ids[:eos_idx]
            
            hypothesis = sp.decode(output_ids)
            all_hypotheses.append(hypothesis)
    
    # BLEU score
    print("\n" + "="*70)
    print("Computing BLEU score...")
    bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
    print(f"BLEU Score: {bleu.score:.2f}")
    print("="*70 + "\n")
    
    if num_samples > 0:
        print("="*70)
        print("SAMPLE TRANSLATIONS")
        print("="*70)
        
        num_to_display = min(num_samples, len(src_data))
        for i in range(num_to_display):
            src_text = sp.decode(src_data[i])
            print(f"\n--- Sample {i+1} ---")
            print(f"Source:     {src_text}")
            print(f"Reference:  {tgt_texts[i]}")
            print(f"Hypothesis: {all_hypotheses[i]}")
        
        print("\n" + "="*70)
    
    return bleu.score


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer on English-French test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoints-path', type=str,
                        help='Path to checkpoints file (auto-detect latest if not provided)')
    parser.add_argument('--model-name', type=str, default="best",
                        help='Name of the model, not including the extension. The path will be constructed')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of sample translations to display')
    parser.add_argument('--batch-size', type=int, default=EVAL_BATCH_SIZE,
                        help='Batch size for inference')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.device("cuda")
    logging.info(f"Using device: {device}")
    
    sp_model_path = SP_MODEL_PATH
    test_src_file = TEST_SRC_FILE
    test_tgt_raw_file = TEST_TGT_RAW_FILE
    checkpoints_dir = CHECKPOINTS_DIR

    if not sp_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {sp_model_path}")
    if not test_src_file.exists():
        raise FileNotFoundError(f"Test source file not found at {test_src_file}")
    if not test_tgt_raw_file.exists():
        raise FileNotFoundError(f"Test target raw file not found at {test_tgt_raw_file}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    vocab_size = sp.get_piece_size()
    logging.info(f"Vocabulary size: {vocab_size}")

    if args.model_name:
        checkpoints_path = checkpoints_dir / f'{args.model_name}.pt'
        if not checkpoints_path.exists():
            raise FileNotFoundError(f"Checkpoints not found at {checkpoints_path}")
    elif args.checkpoints_path:
        checkpoints_path = Path(args.checkpoints_path)
        if not checkpoints_path.exists():
            raise FileNotFoundError(f"Checkpoints not found at {checkpoints_path}")
    else:
        raise ValueError("No model name or checkpoints path provided")

    logging.info(f"Loading checkpoints from: {checkpoints_path}")
    checkpoints = torch.load(str(checkpoints_path), map_location=device, weights_only=False)
    config = checkpoints['model_config']
    epoch = checkpoints['epoch']
    loss = checkpoints['loss']
    logging.info(f"Checkpoints loaded - Epoch: {epoch}, Loss: {loss}")


    d_model = config['d_model']
    num_heads = config['heads']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    dropout = config['dropout']
    max_seq_len = config['max_seq_len']

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoints['model_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f} M)")

    # Evaluate on test set
    bleu_score = evaluate_on_test_set(
        model=model,
        sp_model_path=sp_model_path,
        test_src_file=test_src_file,
        test_tgt_raw_file=test_tgt_raw_file,
        device=device,
        max_seq_len=max_seq_len,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    logging.info(f"Final BLEU Score: {bleu_score:.2f}")




    # parser = argparse.ArgumentParser(
    #     description="Evaluate Transformer on English-French test set",
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # # parser.add_argument('--checkpoints', type=str, default=str(CHECKPOINTS_DIR / "transformer_epoch_1.pt"),
    # #                     help='Path to checkpoints file (auto-detect latest if not provided)')
    # parser.add_argument('--checkpoints-path', type=str,
    #                     help='Path to checkpoints file (auto-detect latest if not provided)')
    # parser.add_argument('--model-name', type=str, default=model_name,
    #                     help='Name of the model, not including the extension. The paath will be constructed')
    # parser.add_argument('--num-samples', type=int, default=5,
    #                     help='Number of sample translations to display')
    # parser.add_argument('--batch-size', type=int, default=EVAL_BATCH_SIZE,
    #                     help='Batch size for inference')

    # parser.add_argument('--max-seq-len', type=int, default=max_seq_len,
                        # help='Maximum sequence length for generation')
    
    # parser.add_argument('--d-model', type=int, default=512,
                        # help='Dimension of model embeddings (must match training)')
    # parser.add_argument('--heads', type=int, default=4,
                        # help='Number of attention heads (must match training)')
    # parser.add_argument('--d-ff', type=int, default=2048,
                        # help='Dimension of feedforward network (must match training)')
    # parser.add_argument('--num-layers', type=int, default=3,
                        # help='Number of encoder/decoder layers (must match training)')
    # parser.add_argument('--dropout', type=float, default=0.1,
                        # help='Dropout rate (must match training)')
    
    # args = parser.parse_args()
    
    # if not torch.cuda.is_available():
    #     raise ValueError("CUDA is not available")
    # device = torch.device("cuda")
    # logging.info(f"Using device: {device}")
    
    # sp_model_path = SP_MODEL_PATH
    # test_src_file = TEST_SRC_FILE
    # test_tgt_raw_file = TEST_TGT_RAW_FILE
    
    # if not sp_model_path.exists():
    #     raise FileNotFoundError(f"SentencePiece model not found at {sp_model_path}")
    # if not test_src_file.exists():
    #     raise FileNotFoundError(f"Test source file not found at {test_src_file}")
    # if not test_tgt_raw_file.exists():
    #     raise FileNotFoundError(f"Test target raw file not found at {test_tgt_raw_file}")
    
    # sp = spm.SentencePieceProcessor()
    # sp.load(str(sp_model_path))
    # vocab_size = sp.get_piece_size()
    # logging.info(f"Vocabulary size: {vocab_size}")
    
    # model = Transformer(
    #     src_vocab_size=vocab_size,
    #     tgt_vocab_size=vocab_size,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     d_ff=d_ff,
    #     num_layers=num_layers,
    #     dropout=dropout
    # ).to(device)
    
    # total_params = sum(p.numel() for p in model.parameters())
    # logging.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f} M)")
    
    # Load checkpoints
    # if args.checkpoints:
    #     checkpoints_path = Path(args.checkpoints)
    # else:
    #     logging.info("No checkpoint specified, searching for latest...")
    #     checkpoints_path = find_latest_checkpoints(CHECKPOINTS_DIR)

    # if args.model_name:
    #     checkpoints_path = checkpoints_dir / f'{args.model_name}.pt'
    #     if not checkpoints_path.exists():
    #         raise FileNotFoundError(f"Checkpoints not found at {checkpoints_path}")
    # elif args.checkpoints_path:
    #     checkpoints_path = Path(args.checkpoints_path)
    #     if not checkpoints_path.exists():
    #         raise FileNotFoundError(f"Checkpoints not found at {checkpoints_path}")
    # else:
    #     raise ValueError("No model name or checkpoints path provided")

    
    # model = load_checkpoints(checkpoints_path, model, device)
    
    # # Evaluate on test set
    # bleu_score = evaluate_on_test_set(
    #     model=model,
    #     sp_model_path=sp_model_path,
    #     test_src_file=test_src_file,
    #     test_tgt_raw_file=test_tgt_raw_file,
    #     device=device,
    #     max_seq_len=max_seq_len,
    #     batch_size=args.batch_size,
    #     num_samples=args.num_samples
    # )
    
    # logging.info(f"Final BLEU Score: {bleu_score:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    main()

