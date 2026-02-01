#!/usr/bin/env python3
"""
Romanian TTS Inference with HuggingFace Model Integration

This script demonstrates how to use pre-trained Romanian Matcha-TTS models
from HuggingFace with the main repository's inference capabilities.

Usage:
    python inference_hf.py --text "Bună ziua!" --model bas_950
    python inference_hf.py --file sample_texts.txt --model sgs_10 --output_dir ./samples
"""

import argparse
import os
import sys
import torch
import soundfile as sf
from pathlib import Path
import datetime as dt
from tqdm.auto import tqdm

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

# HuggingFace model loader
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "Ro-Matcha-TTS" / "src"))
    from model_loader import ModelLoader
    HF_INTEGRATION = True
except ImportError:
    print("Warning: HuggingFace integration not available. Place Ro-Matcha-TTS repository alongside this one.")
    HF_INTEGRATION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Romanian TTS with HuggingFace models")

    # Text input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text to synthesize")
    group.add_argument("--file", type=str, help="File with texts to synthesize (format: filename|text|speaker)")

    # Model selection
    parser.add_argument("--model", type=str, default="bas_950",
                       choices=["swara", "bas_10", "bas_950", "sgs_10", "sgs_950"],
                       help="Model to use")
    parser.add_argument("--hf_repo", type=str, default="../Ro-Matcha-TTS",
                       help="Path to HuggingFace model repository or repo ID")

    # Synthesis parameters
    parser.add_argument("--n_timesteps", type=int, default=50, help="Number of ODE solver steps")
    parser.add_argument("--length_scale", type=float, default=0.95, help="Speaking rate control")
    parser.add_argument("--temperature", type=float, default=0.667, help="Sampling temperature")

    # Output
    parser.add_argument("--output_dir", type=str, default="./hf_samples", help="Output directory")
    parser.add_argument("--output_name", type=str, help="Output filename (for single text mode)")

    return parser.parse_args()

def load_models_from_hf(hf_repo: str, model: str):
    """Load models using HuggingFace integration"""
    if not HF_INTEGRATION:
        raise ImportError("HuggingFace integration not available")

    print(f"Loading models from HuggingFace repo: {hf_repo}")

    # Initialize model loader
    loader = ModelLoader.from_pretrained(hf_repo)
    model_info = loader.load_models(model=model, device=device)

    print(f"Using model: {model_info['model_name']}")
    print(f"Description: {model_info['model_info']['description']}")
    print(f"Training data: {model_info['model_info'].get('training_data', 'N/A')}")
    print(f"Device: {model_info['device']}")

    # Load TTS model
    print("Loading TTS model...")
    tts_model = MatchaTTS.load_from_checkpoint(model_info['model_path'], map_location=device)
    tts_model.eval()

    # Load vocoder
    print("Loading vocoder...")
    h = AttrDict(v1)
    vocoder = HiFiGAN(h).to(device)
    vocoder_checkpoint = torch.load(model_info['vocoder_path'], map_location=device)
    vocoder.load_state_dict(vocoder_checkpoint['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    denoiser = Denoiser(vocoder, mode='zeros')

    return {
        'tts_model': tts_model,
        'vocoder': vocoder,
        'denoiser': denoiser,
        'config': model_info['config'],
        'inference_params': model_info['inference_params']
    }

def process_text(text: str):
    """Process Romanian text for synthesis"""
    x = torch.tensor(
        intersperse(text_to_sequence(text, ['romanian_cleaners'])[0], 0),
        dtype=torch.long,
        device=device
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())

    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }

@torch.inference_mode()
def synthesize(text: str, models: dict, args):
    """Synthesize speech from text"""
    text_processed = process_text(text)
    start_t = dt.datetime.now()

    # Use inference parameters from model config or command line
    params = models['inference_params']
    n_timesteps = args.n_timesteps if args.n_timesteps != 50 else params['n_timesteps']
    temperature = args.temperature if args.temperature != 0.667 else params['temperature']
    length_scale = args.length_scale if args.length_scale != 0.95 else params['length_scale']

    # Synthesis
    output = models['tts_model'].synthesise(
        text_processed['x'],
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        length_scale=length_scale
    )

    # Convert to waveform
    mel = output['mel']
    audio = models['vocoder'](mel).clamp(-1, 1)
    audio = models['denoiser'](audio.squeeze(0), strength=0.00025).cpu().squeeze()

    # Add timing info
    end_t = dt.datetime.now()
    rtf = (end_t - start_t).total_seconds() / (audio.shape[0] / models['config']['sample_rate'])

    return {
        'audio': audio.numpy(),
        'sample_rate': models['config']['sample_rate'],
        'text': text,
        'rtf': rtf,
        **text_processed
    }

def parse_filelist(filelist_path: str, split_char="|"):
    """Parse filelist for batch processing"""
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # Load models
    try:
        models = load_models_from_hf(args.hf_repo, args.model)
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        return

    # Single text mode
    if args.text:
        print(f"\nSynthesizing: '{args.text}'")

        try:
            result = synthesize(args.text, models, args)

            # Save audio
            if args.output_name:
                output_file = output_dir / args.output_name
                if not output_file.suffix:
                    output_file = output_file.with_suffix('.wav')
            else:
                output_file = output_dir / f"sample_{args.model}_single.wav"

            sf.write(output_file, result['audio'], result['sample_rate'])
            print(f"✓ Saved: {output_file}")
            print(f"  RTF: {result['rtf']:.4f}")
            print(f"  Phonemes: {result['x_phones']}")

        except Exception as e:
            print(f"✗ Synthesis failed: {e}")

    # Batch file mode
    elif args.file:
        print(f"\nProcessing file: {args.file}")

        try:
            filelist = parse_filelist(args.file)
            print(f"Found {len(filelist)} texts to synthesize")

            rtf_values = []
            for i, line in enumerate(tqdm(filelist, desc="Synthesizing")):
                if len(line) >= 3:
                    filename, text, file_model = line[0], line[1], line[2]
                    # For backward compatibility, map speaker names to models
                    if file_model in ["BAS", "SGS"]:
                        file_model = f"{file_model.lower()}_950"  # Use high-quality version
                elif len(line) >= 2:
                    filename, text = line[0], line[1]
                    file_model = args.model
                else:
                    print(f"Skipping invalid line {i+1}: {line}")
                    continue

                try:
                    result = synthesize(text, models, args)

                    # Use original filename or create new one
                    if filename.endswith('.wav'):
                        output_file = output_dir / filename
                    else:
                        output_file = output_dir / f"{filename}.wav"

                    sf.write(output_file, result['audio'], result['sample_rate'])
                    rtf_values.append(result['rtf'])

                except Exception as e:
                    print(f"Failed to synthesize line {i+1}: {e}")
                    continue

            # Save RTF statistics
            if rtf_values:
                rtf_stats = {
                    'mean': sum(rtf_values) / len(rtf_values),
                    'min': min(rtf_values),
                    'max': max(rtf_values),
                    'count': len(rtf_values)
                }

                print(f"\n📊 Performance Statistics:")
                print(f"  Average RTF: {rtf_stats['mean']:.4f}")
                print(f"  Min RTF: {rtf_stats['min']:.4f}")
                print(f"  Max RTF: {rtf_stats['max']:.4f}")
                print(f"  Processed: {rtf_stats['count']} samples")

        except Exception as e:
            print(f"✗ Batch processing failed: {e}")

if __name__ == "__main__":
    main()