# ================================================================================
import argparse
import datetime as dt
import os
from pathlib import Path

import IPython.display as ipd
import numpy as np
import pandas as pd
from regex import F
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

# ================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPEAKER = None
SAVE_DIR = None

model = None
vocoder = None
denoiser = None

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--n_timesteps",
        type=int,
        required=False,
        default=50,
        help="Number of ODE Solver steps",
    )

    parser.add_argument(
        "--length_scale",
        type=float,
        required=False,
        default=0.95,
        help="Length scale for the diffusion process",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.667,
        help="Temperature for sampling",
    )

    # Experiment settings
    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help='path to a file with texts to synthesize'
    )
    parser.add_argument(
        '-c', '--checkpoint',
        type=str,
        required=True,
        help='Name of the checkpoint file to use (e.g: matcha-tts-BAS.ckpt)'
    )

    args = parser.parse_args()
    return args

def load_model(checkpoint_path):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model

def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def count_params(x):
    return f"{sum(p.numel() for p in x.parameters()):,}"

@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['romanian_cleaners'])[0], 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text, args, spks=None):
    global model

    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'],
        text_processed['x_lengths'],
        n_timesteps=args.n_timesteps,
        temperature=args.temperature,
        spks=spks,
        length_scale=args.length_scale
    )
    # merge everything to one dict
    output.update({'start_t': start_t, **text_processed})
    return output

@torch.inference_mode()
def to_waveform(mel, vocoder):
    global denoiser

    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()

def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    # np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}', output['waveform'], 22050, 'PCM_24')

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def main(args):
    CKPT_DIR = "/workspace/local/checkpoints"
    MATCHA_CHECKPOINT = CKPT_DIR + f"/matcha-tts/{args.checkpoint}"
    HIFIGAN_CHECKPOINT = CKPT_DIR + "/hifigan_univ_v1"

    # args.checkpoint is a file path, get the file name without the extension
    _, speaker_id, train_epochs = os.path.basename(args.checkpoint).split('.')[0].split('-')

    print(f"Using Matcha checkpoint: {speaker_id} {train_epochs}")


    OUTPUT_FOLDER = "/workspace/local/samples/27_aug_2025_matcha"
    OUTPUT_FOLDER += f"/{speaker_id}/{train_epochs}"

    print(f"Output folder: {OUTPUT_FOLDER}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # exit(0)


    print(f"Using Matcha checkpoint: {MATCHA_CHECKPOINT}")
    print(f"Using HiFi-GAN checkpoint: {HIFIGAN_CHECKPOINT}")

    global model, vocoder, denoiser

    # # Create output directory
    # if 'bas' in str(args.checkpoint).lower():
    #     SPEAKER = 'bas'
    #     OUTPUT_FOLDER += f"/{args.checkpoint}"
    #     if not os.path.exists(OUTPUT_FOLDER):
    #         os.makedirs(OUTPUT_FOLDER)
    # elif 'sgs' in str(args.checkpoint).lower():
    #     SPEAKER = 'sgs'
    #     OUTPUT_FOLDER += f"/{args.checkpoint}"
    #     if not os.path.exists(OUTPUT_FOLDER):
    #         os.makedirs(OUTPUT_FOLDER)
    # elif 'flo' in str(args.checkpoint).lower():
    #     SPEAKER = 'flo'
    #     OUTPUT_FOLDER += f"/{args.checkpoint}"
    #     if not os.path.exists(OUTPUT_FOLDER):
    #         os.makedirs(OUTPUT_FOLDER)
    # else:
    #     raise ValueError(f"Unknown model type in checkpoint path: {args.checkpoint}")
    # print(f"Saving to {OUTPUT_FOLDER}")

    print('Initializing model...')
    model = load_model(MATCHA_CHECKPOINT)
    print(f"Model loaded! Parameter count: {count_params(model)}")

    print('Initializing HiFi-GAN...')
    print(HIFIGAN_CHECKPOINT)
    vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
    denoiser = Denoiser(vocoder, mode='zeros')

    print(f"Reading texts from {args.file}...")
    filelist = parse_filelist(args.file, split_char='|')

    # rtf_values = []
    with torch.no_grad():
        for i, line in enumerate(tqdm(filelist, desc="Synthesizing")):

            filepath, text, speaker = line[0], line[1], line[2]

            output = synthesise(text, args=args)
            output['waveform'] = to_waveform(output['mel'], vocoder)
            # rtf_values.append(output["rtf"])

            # Filepath is a full path, extract the base path
            base_name = os.path.basename(filepath)
            save_to_folder(base_name, output, OUTPUT_FOLDER)

    # print('Done. Check out `out` folder for samples.')
    # rtf_df = pd.DataFrame(rtf_values, columns=['RTF'])
    # csv_file = os.path.join(OUTPUT_FOLDER, 'rtf_values.csv')
    # rtf_df.to_csv(csv_file, index=False)

    # stats = rtf_df.describe().loc[['mean', 'max', 'min', 'std']]
    # stats.index = ['Average RTF', 'Max RTF', 'Min RTF', 'Standard Deviation RTF']
    # stats_csv_file = os.path.join(OUTPUT_FOLDER, 'rtf_stats.csv')
    # stats.to_csv(stats_csv_file)

    # print(f"RTF values saved to {csv_file}")
    # print(f"RTF statistics saved to {stats_csv_file}")

    # print(f"RTF values saved to {csv_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)

