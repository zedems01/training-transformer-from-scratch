import os
import sentencepiece as spm
import logging
import argparse
from src.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    VOCAB_SIZE,
    MAX_SENTENCES
)


def train_sentencepiece(input_file, model_prefix, vocab_size):
    """
    Train SentencePiece model on a combined text file of source + target.
    """
    logging.info(f"Training SentencePiece model with vocab size {vocab_size}")
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        "--model_type=bpe --minloglevel=2 "
        "--user_defined_symbols=<PAD>,<BOS>,<EOS> "
        "--unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3 "
        "--unk_piece=<UNK> --pad_piece=<PAD> --bos_piece=<BOS> --eos_piece=<EOS>"
    )


def encode_with_sentencepiece(sp_model_path, input_file, output_file):
    """
    Encode a raw text file using the trained SentencePiece model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            pieces = sp.encode(line.strip(), out_type=str)
            fout.write(" ".join(pieces) + "\n")


def split_parallel_data(
    raw_dir,
    output_dir,
    train_ratio,
    val_ratio,
    max_sentences
):
    """
    Split the combined parallel data into train/validation/test sets.
    
    Args:
        max_sentences: Maximum number of sentence pairs to use (default: 100k pairs)
    """
    logging.info(f"Creating train/validation/test splits (max {max_sentences:,} sentences)...")
    os.makedirs(output_dir, exist_ok=True)
    
    # parallel files from our downloaded dataset
    parallel_files = [
        ('europarl-v7.fr-en.en', 'europarl-v7.fr-en.fr'),
        ('commoncrawl.fr-en.en', 'commoncrawl.fr-en.fr'),
        ('news-commentary-v9.fr-en.en', 'news-commentary-v9.fr-en.fr')
    ]
    
    # collect all parallel sentences
    en_sentences, fr_sentences = [], []
    for en_file, fr_file in parallel_files:
        en_path = os.path.join(raw_dir, en_file)
        fr_path = os.path.join(raw_dir, fr_file)

        if not (os.path.exists(en_path) and os.path.exists(fr_path)):
            logging.warning(f"Warning: Could not find {en_file} or {fr_file}")
            continue
            
        with open(en_path, 'r', encoding='utf-8') as en_f, \
             open(fr_path, 'r', encoding='utf-8') as fr_f:
            for en_line, fr_line in zip(en_f, fr_f):
                en_sentences.append(en_line.strip())
                fr_sentences.append(fr_line.strip())
                if len(en_sentences) >= max_sentences:
                    break
        if len(en_sentences) >= max_sentences:
            break
    
    # trim to max_sentences if needed
    if len(en_sentences) > max_sentences:
        en_sentences = en_sentences[:max_sentences]
        fr_sentences = fr_sentences[:max_sentences]
    # calculate split sizes
    total_size = len(en_sentences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    # create splits
    splits = {
        'train': (0, train_size),
        'valid': (train_size, train_size + val_size),
        'test': (train_size + val_size, total_size)
    }
    
    # write splits to files
    for split_name, (start, end) in splits.items():
        en_output = os.path.join(output_dir, f'{split_name}.en')
        fr_output = os.path.join(output_dir, f'{split_name}.fr')
        
        with open(en_output, 'w', encoding='utf-8') as en_f, \
             open(fr_output, 'w', encoding='utf-8') as fr_f:
            for en, fr in zip(en_sentences[start:end], fr_sentences[start:end]):
                en_f.write(en + '\n')
                fr_f.write(fr + '\n')
        
        logging.info(f"Created {split_name} split with {end-start} sentence pairs\n")


def process_data(
    raw_dir="./src/data/raw",
    processed_dir="./src/data/processed",
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    vocab_size=VOCAB_SIZE,
    max_sentences=MAX_SENTENCES
):
    """
    Complete data processing pipeline:
    1. Split data into train/valid/test
    2. Train SentencePiece model
    3. Encode all splits with SentencePiece
    
    Args:
        raw_dir: Directory to save the raw dataset
        processed_dir: Directory to save the processed dataset
        vocab_size: Vocabulary size
        max_sentences: Maximum number of sentence pairs to use (default: 100k pairs)
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # create data splits
    split_parallel_data(
        raw_dir,
        processed_dir,
        train_ratio,
        val_ratio,
        max_sentences
    )
    
    # combine all training data for BPE training
    combined_file = os.path.join(processed_dir, "combined_corpus.txt")
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for lang in ['en', 'fr']:
            train_file = os.path.join(processed_dir, f'train.{lang}')
            with open(train_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    
    # train SentencePiece model
    model_prefix = os.path.join(processed_dir, "sp_bpe")
    train_sentencepiece(combined_file, model_prefix, vocab_size)
    
    # encode all splits
    for split in ['train', 'valid', 'test']:
        for lang in ['en', 'fr']:
            input_file = os.path.join(processed_dir, f'{split}.{lang}')
            output_file = os.path.join(processed_dir, f'{split}.{lang}.bpe')
            encode_with_sentencepiece(f"{model_prefix}.model", input_file, output_file)
    
    # clean up combined corpus file
    os.remove(combined_file)
    logging.info("Data processing complete!\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    parser = argparse.ArgumentParser(
        description='Process WMT14 English-French dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--raw-dir', type=str, default='./src/data/raw',
                        help='Directory to save the raw dataset')
    parser.add_argument('--processed-dir', type=str, default='./src/data/processed',
                        help='Directory to save the processed dataset')
    parser.add_argument('--vocab-size', type=int, default=VOCAB_SIZE,
                        help='Vocabulary size')
    parser.add_argument('--max-sentences', type=int, default=MAX_SENTENCES,
                        help='Maximum number of sentence pairs to use')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO,
                        help='Training ratio')
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO,
                        help='Validation ratio')
    args = parser.parse_args()
    process_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        vocab_size=args.vocab_size,
        max_sentences=args.max_sentences,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    