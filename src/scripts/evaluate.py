import json
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

from src.scripts.utils import create_pad_mask, greedy_decode

from src.scripts.transformer import Transformer
from src.config import (
    CHECKPOINTS_DIR,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    EVAL_BATCH_SIZE,
    SP_MODEL_PATH,
    TEST_SRC_FILE,
    TEST_TGT_RAW_FILE,
)



class TranslationTestDataset(Dataset):
    def __init__(self, src_file, tgt_raw_file, sp_model, max_len):
        super().__init__()
        self.sp = sp_model
        self.max_len = max_len
        self.src_data = []
        self.tgt_texts = []
        
        with src_file.open('r', encoding='utf-8') as src_f, \
             tgt_raw_file.open('r', encoding='utf-8') as tgt_raw_f:
            
            filtered_count = 0
            for src_line, tgt_raw_line in zip(src_f, tgt_raw_f):
                src_tokens = src_line.strip().split()
                
                if len(src_tokens) <= self.max_len - 2:
                    # convert to tensors
                    src_indices = [self.sp.piece_to_id(token) for token in src_tokens]
                    self.src_data.append(torch.tensor(src_indices, dtype=torch.long))
                    self.tgt_texts.append(tgt_raw_line.strip())
                else:
                    filtered_count += 1
        
        logging.info(f"Loaded {len(self.src_data)} test samples.")
        logging.info(f"Filtered out {filtered_count} sequences exceeding max_len={self.max_len}")

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        """
        Returns:
          src_data: Source token ID sequence
          tgt_text: Original target text (reference)
        """
        return self.src_data[idx], self.tgt_texts[idx]

    @staticmethod
    def collate_fn(batch):
        """
        Collate batch with padding
        """
        src_tensors = [item[0] for item in batch]
        tgt_texts = [item[1] for item in batch]
        src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=PAD_ID)
        
        return src_padded, tgt_texts


def evaluate_on_test_set(
    model,
    sp_model_path,
    test_src_file,
    test_tgt_raw_file,
    results_file,
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
        results_file: Path to save results file
        device: Device to run on
        max_seq_len: Maximum sequence length
        batch_size: Batch size for inference
        num_samples: Number of sample translations to display
    
    Returns:
        BLEU score
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    test_dataset = TranslationTestDataset(
        test_src_file, test_tgt_raw_file, sp, max_seq_len
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TranslationTestDataset.collate_fn,
        pin_memory=True
    )
    
    model.eval()
    all_hypotheses = []
    all_references = []
    
    logging.info("Generating translations...")
    for padded_src, batch_tgt_texts in tqdm(test_loader, desc="Translating"):
        padded_src = padded_src.to(device)
        all_references.extend(batch_tgt_texts)
        
        src_mask = create_pad_mask(padded_src, pad_idx=PAD_ID)
        
        with torch.no_grad():
            output = greedy_decode(
                model, padded_src, src_mask, 
                max_len=max_seq_len, device=device,
                bos_id=BOS_ID, eos_id=EOS_ID
            )
        
        output_cpu = output.cpu().tolist()
        
        for output_ids in output_cpu:
            if output_ids and output_ids[0] == BOS_ID:
                output_ids = output_ids[1:]
            if EOS_ID in output_ids:
                eos_idx = output_ids.index(EOS_ID)
                output_ids = output_ids[:eos_idx]
            
            hypothesis = sp.decode(output_ids)
            all_hypotheses.append(hypothesis)
    
    print("\n" + "="*70)
    print("Computing BLEU score...")
    bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
    print(f"BLEU Score: {bleu.score:.2f}")
    print("="*70 + "\n")
    
    results = {
        "bleu_score": bleu.score,
        "signature": str(bleu),
        "samples": []
    }
    
    num_to_display = min(num_samples, len(test_dataset))
    for i in range(num_to_display):
        src_tensor, _ = test_dataset[i]
        src_text = sp.decode(src_tensor.tolist())
        
        results["samples"].append({
            "id": i + 1,
            "source": src_text,
            "reference": all_references[i],
            "hypothesis": all_hypotheses[i]
        })

    if num_samples > 0:
        print("="*70)
        print("SAMPLE TRANSLATIONS")
        print("="*70)
        for sample in results["samples"]:
            print(f"\n--- Sample {sample['id']} ---")
            print(f"Source:     {sample['source']}")
            print(f"Reference:  {sample['reference']}")
            print(f"Hypothesis: {sample['hypothesis']}")
        print("\n" + "="*70)

    try:
        with results_file.open('w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"Results successfully saved to {results_file}")
    except IOError as e:
        logging.error(f"Error saving results to {results_file}: {e}")
    
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
    parser.add_argument('--results-file', type=str,
                        help='Path to results file, not including the extension')
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

    if args.results_file:
        results_file = checkpoints_dir / f"{args.results_file}_{args.model_name}.json"
    else:
        results_file = checkpoints_dir / f"results_{args.model_name}.json"

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
        num_samples=args.num_samples,
        results_file=results_file
    )
    
    logging.info(f"Final BLEU Score: {bleu_score:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    main()

