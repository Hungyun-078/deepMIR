#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse, warnings, multiprocessing as mp
from pathlib import Path
from typing import Tuple, List
import numpy as np
from tqdm import tqdm
import librosa, dill

# --------------------------------
# 參數預設（在同一路徑跑即可）
# --------------------------------
dataset_root = Path('/tmp2/mir/data/hw1')
ARTISTS_DIR    = dataset_root / "artist20" / "train_val"          # 原始音檔根目錄
OUT_DIR        = dataset_root / "song_total_data"  # 特徵輸出目錄（自動建立）

AUDIO_EXTS = ".mp3"

# --------------------------------
# 小工具
# --------------------------------
def sanitize(name: str) -> str:
    return re.sub(r'[\\/:"*?<>|]+', "_", name)

def pkl_name(artist:str, album:str, song:str) -> str:
    return f"{sanitize(artist)}__{sanitize(album)}__{sanitize(song)}.pkl"

def load_audio(path: str, sr: int, use_torchaudio: bool = False) -> np.ndarray:
    if not use_torchaudio:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y
    try:
        import torchaudio
        wav, orig_sr = torchaudio.load(path)         
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        return wav.squeeze(0).numpy()
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y

def compute_logmel(
    song_path: str,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_seconds: int = 120, #cut 120s
    crop_mode: str = "center", # from center split
    seed: int = 42,
    use_torchaudio: bool = False,
) -> np.ndarray:
    
    y = load_audio(song_path, sr=sr, use_torchaudio=use_torchaudio)

    if max_seconds and max_seconds > 0:
        max_len = int(max_seconds * sr)
        if len(y) > max_len:
            if crop_mode == "center":
                start = (len(y) - max_len) // 2
            elif crop_mode == "random":
                rng = np.random.RandomState(seed)
                start = int(rng.randint(0, len(y) - max_len))
            else:
                start = 0
            y = y[start:start + max_len]
    
    # Compute log-mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, power=2.0
    ) 
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def process_one(args) -> Tuple[str, bool, str]:
    (artist, album, song, sr, n_mels, n_fft, hop, max_seconds,
     crop_mode, seed, out_dir, force_rebuild, save_format, use_ta) = args
    try:
        song_path = ARTISTS_DIR / artist / album / song
        if song_path.suffix.lower() not in AUDIO_EXTS:
            return (str(song_path), False, "skip_non_audio")

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        base = pkl_name(artist, album, song)
        out_path = out_dir / (base if save_format == "pkl" else Path(base).with_suffix(".npz").name)

        if out_path.exists() and not force_rebuild:
            return (str(song_path), True, "cached")

        logmel = compute_logmel(
            str(song_path), sr, n_mels, n_fft, hop, max_seconds, crop_mode, seed, use_ta
        )

        data = {
            "artist": artist,
            "album": album,
            "song": song,
            "sr": sr, "n_mels": n_mels, "n_fft": n_fft, "hop_length": hop,
            "max_seconds": max_seconds, "crop_mode": crop_mode
        }

        if save_format == "pkl":
            with open(out_path, "wb") as fp:
                dill.dump({"logmel": logmel, **data}, fp, protocol=dill.HIGHEST_PROTOCOL)
        else:
            np.savez_compressed(out_path, logmel=logmel, **data)

        return (str(song_path), True, "ok")
    except Exception as e:
        return (str(ARTISTS_DIR / artist / album / song), False, str(e))

def crawl_items() -> List[tuple]:
    items = []
    for artist in os.listdir(ARTISTS_DIR):
        ap = ARTISTS_DIR / artist
        if not ap.is_dir(): 
            continue
        for album in os.listdir(ap):
            alp = ap / album
            if not alp.is_dir():
                continue
            for song in os.listdir(alp):
                if (alp / song).suffix.lower() in AUDIO_EXTS:
                    items.append((artist, album, song))
    return items

def run_precompute(
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop: int = 512,
    max_seconds: int = 120,
    crop_mode: str = "center",
    seed: int = 42,
    num_worker: int = 4,
    force_rebuild: bool = False,
    save_format: str = "pkl",        
    use_torchaudio: bool = False,
    chunksize: int = 1,
):
    items = crawl_items()
    print(f"[Precompute] root={ARTISTS_DIR}  total_files={len(items)}  out={OUT_DIR}")
    if not items:
        print("No audio files in artists/. Expected: artists/<artist>/<album>/*.audio")
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tasks = [
        (artist, album, song, sr, n_mels, n_fft, hop, max_seconds,
         crop_mode, seed, str(OUT_DIR), force_rebuild, save_format, use_torchaudio)
        for (artist, album, song) in items
    ]

    ok = fail = 0
    with mp.Pool(processes=num_worker) as pool:
        for path, succ, msg in tqdm(
            pool.imap_unordered(process_one, tasks, chunksize=chunksize),
            total=len(tasks), desc="wave2spec"
        ):
            if succ: ok += 1
            else:
                fail += 1
                tqdm.write(f"[WARN] {path} -> {msg}")
    print(f"[Precompute] OK={ok}  FAIL={fail}  saved_to={OUT_DIR}")

def main():
    parser = argparse.ArgumentParser("Audio → log-mel precompute (local, same folder)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--max_seconds", type=int, default=120)
    parser.add_argument("--crop_mode", type=str, default="center", choices=["center","random","none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--save_format", type=str, default="pkl", choices=["pkl","npz"])
    parser.add_argument("--use_torchaudio", action="store_true")
    parser.add_argument("--chunksize", type=int, default=1)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_precompute(
        sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft, hop=args.hop_length,
        max_seconds=args.max_seconds, crop_mode=args.crop_mode, seed=args.seed,
        num_worker=args.num_worker, force_rebuild=args.force_rebuild,
        save_format=args.save_format, use_torchaudio=args.use_torchaudio,
        chunksize=args.chunksize
    )

if __name__ == "__main__":
    main()