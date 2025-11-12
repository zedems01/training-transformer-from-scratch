import logging
import argparse
from pathlib import Path
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from src.scripts.transformer import Transformer

from src.scripts.utils import create_pad_mask, greedy_decode
from src.config import (
    SP_MODEL_PATH,
    CHECKPOINTS_DIR,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    INFERENCE_BATCH_SIZE
)


class Translator:
    """
    Translator class for inference with a trained Transformer model.
    """
    def __init__(self, model, sp_model, device, max_len=512):
        self.model = model
        self.sp = sp_model
        self.device = device
        self.max_len = max_len
        self.model.eval()
    
    def translate_sentence(self, sentence: str) -> str:
        # encode sentence to BPE tokens
        bpe_tokens = self.sp.encode(sentence, out_type=str)
        token_ids = [self.sp.piece_to_id(token) for token in bpe_tokens]
        src = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        src_mask = create_pad_mask(src, pad_idx=PAD_ID)
        
        # model translation
        output = greedy_decode(
            self.model, src, src_mask,
            max_len=self.max_len, device=self.device,
            bos_id=BOS_ID, eos_id=EOS_ID
        )
        
        # decode output
        output_ids = output[0].cpu().tolist()
        if output_ids[0] == BOS_ID:
            output_ids = output_ids[1:]
        if EOS_ID in output_ids:
            eos_idx = output_ids.index(EOS_ID)
            output_ids = output_ids[:eos_idx]
        translation = self.sp.decode(output_ids)
        
        return translation
    
    def translate_sentences(self, sentences: List[str], batch_size: int) -> List[str]:
        """
        Translate multiple sentences in batches.
        """
        translations = []
        
        # process in batches
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            batch_token_ids = []
            for sentence in batch_sentences:
                bpe_tokens = self.sp.encode(sentence, out_type=str)
                token_ids = [self.sp.piece_to_id(token) for token in bpe_tokens]
                batch_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
            
            padded_src = pad_sequence(
                batch_token_ids,
                batch_first=True,
                padding_value=PAD_ID
            )
            padded_src = padded_src.to(self.device)
            
            src_mask = create_pad_mask(padded_src, pad_idx=PAD_ID)
            
            # model translations
            outputs = greedy_decode(
                self.model, padded_src, src_mask,
                max_len=self.max_len, device=self.device,
                bos_id=BOS_ID, eos_id=EOS_ID
            )
            
            outputs_cpu = outputs.cpu().tolist()
            
            # decode outputs
            for output_ids in outputs_cpu:
                if output_ids[0] == BOS_ID:
                    output_ids = output_ids[1:]
                if EOS_ID in output_ids:
                    eos_idx = output_ids.index(EOS_ID)
                    output_ids = output_ids[:eos_idx]
                translation = self.sp.decode(output_ids)
                translations.append(translation)
        
        return translations


def interactive_mode(translator: Translator):
    """
    Interactive translation mode in CLI.
    """
    print("\n" + "="*70)
    print("Interactive Translation Mode")
    print("="*70)
    print("Enter sentences to translate (English -> French)")
    print("Type 'quit', 'q', or 'exit' to stop")
    print("="*70 + "\n")
    
    while True:
        try:
            sentence = input("English:  ").strip()
            
            if sentence.lower() in ['quit', 'exit', 'q']:
                print("exiting interactive mode... catch u next time!")
                break
            if not sentence:
                continue
            
            translation = translator.translate_sentence(sentence)
            print(f"French:  {translation}\n")
        
        except KeyboardInterrupt:
            print("\n\nexiting interactive mode... catch u next time!")
            break
        except Exception as e:
            print(f"Error: {e}")


def translate_from_file(translator: Translator, batch_size: int, input_file: str, output_file: str=None):
    """
    Translate sentences from a file.
    
    Args:
        translator: Translator instance
        input_file: Path to input file, one sentence per line
        output_file: Path to output file
        batch_size: Batch size for processing
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if output_file:
        output_path = Path(output_file)
        logging.info(f"Translations will be saved to {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Translating {len(sentences)} sentences from {input_file}")
    translations = translator.translate_sentences(sentences, batch_size=batch_size)
    
    # write to output file if specified
    if output_file:
        with open(output_path, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        logging.info(f"Translations saved to {output_file}")
    else:
        print("\n" + "="*70)
        print("TRANSLATIONS")
        print("="*70 + "\n")
        for src, tgt in zip(sentences, translations):
            print(f"Source:      {src}")
            print(f"Translation: {tgt}")
            print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Translate English sentences to French using trained Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-name', type=str, default="best",
                        help='Name of the model. Use if model saved in the default checkpoints directory.\
                        Use --checkpoints-path otherwise')
    parser.add_argument('--checkpoints-path', type=str,
                        help='Path to checkpoints file')
    parser.add_argument('--sentence', type=str,
                        help='Single sentence to translate')
    parser.add_argument('--input-file', type=str,
                        help='Input file with sentences to translate (one per line)')
    parser.add_argument('--output-file', type=str,
                        help='Output file for translations (only with --input-file)')
    parser.add_argument('--batch-size', type=int, default=INFERENCE_BATCH_SIZE,
                        help='Batch size for file translation')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive translation mode')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.device("cuda")
    logging.info(f"Using device: {device}")
    
    sp_model_path = SP_MODEL_PATH
    checkpoints_dir = CHECKPOINTS_DIR

    if  not sp_model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {sp_model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    vocab_size = sp.get_piece_size()
    logging.info(f"Vocabulary size: {vocab_size}")

    if args.model_name:
        base_model_name = args.model_name.removesuffix('.pt').removesuffix('.pth')
        checkpoints_path = checkpoints_dir / f'{base_model_name}.pt'
        if not checkpoints_path.exists():
            raise FileNotFoundError(f"Checkpoints not found at {checkpoints_path}")
    elif args.checkpoints_path:
        checkpoints_path = Path(args.checkpoints_path)
        if not checkpoints_path.suffix:
            checkpoints_path = checkpoints_path.with_suffix('.pt')
        if not checkpoints_path.exists():
            raise FileNotFoundError(f"Checkpoints not found at {checkpoints_path}")
    else:
        raise ValueError("No model name or checkpoints path provided")

    logging.info(f"Loading model from: {checkpoints_path}")
    checkpoints = torch.load(str(checkpoints_path), map_location=device, weights_only=False)
    config = checkpoints['model_config']
    logging.info(f"Model loaded successfully!")
    
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(checkpoints['model_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f} M)")
    
    translator = Translator(model, sp, device, max_len=config['max_seq_len'])
    
    if args.sentence:
        logging.info("Translating single sentence...")
        translation = translator.translate_sentence(args.sentence)
        print("\n" + "="*70)
        print(f"Source:      {args.sentence}")
        print(f"Translation: {translation}")
        print("="*70 + "\n")
    elif args.input_file:
        translate_from_file(
            translator, 
            args.batch_size,
            args.input_file, 
            args.output_file
        )
    elif args.interactive:
        interactive_mode(translator)
    else:
        print("No input specified. Use --sentence, --input-file, or --interactive flag.")
        print("Starting interactive mode by default...")
        interactive_mode(translator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    main()

