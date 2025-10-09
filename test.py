#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import librosa
import dill

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN_Model(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 128, hidden_size: int = 128):
        super(CRNN_Model, self).__init__()
        
        # CNN for feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d((2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.15),
            nn.MaxPool2d((2, 2))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d((2, 2))
        )
        
        # RNN for temporal modeling
        self.rnn_input_size = 256 * (n_mels // 8)
        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape 
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  
        x = x.reshape(b, w, c * h)  
        
        # RNN
        x, _ = self.gru(x) 
        
        # Global average pooling over time
        x = x.mean(dim=1) 
        
        # Classifier
        x = self.fc(x)
        return x

# ================================
# Audio Preprocessing
# ================================
def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def compute_logmel(
    audio_path: str,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_seconds: int = 120,
) -> np.ndarray:
    # Load audio
    y = load_audio(audio_path, sr=sr)
    
    # Crop to max_seconds
    if max_seconds and max_seconds > 0:
        max_len = int(max_seconds * sr)
        if len(y) > max_len:
            start = (len(y) - max_len) // 2
            y = y[start:start + max_len]
    
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, power=2.0
    )
    
    # to dB
    S_db = librosa.power_to_db(S, ref=np.max)
    
    return S_db.astype(np.float32)

def preprocess_audio(audio_path: str, max_frames: int = 3750) -> torch.Tensor:

    # Compute log-mel spectrogram
    logmel = compute_logmel(audio_path)
    
    # fixed length
    n_mels, T = logmel.shape
    
    if T > max_frames:
        start = (T - max_frames) // 2
        logmel = logmel[:, start:start + max_frames]
    elif T < max_frames:
        pad_width = max_frames - T
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        logmel = np.pad(logmel, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    # Normalize (z-score)
    mean = logmel.mean()
    std = logmel.std() + 1e-8
    logmel = (logmel - mean) / std
    
    logmel = torch.FloatTensor(logmel).unsqueeze(0).unsqueeze(0)
    
    return logmel

# ================================
def predict_audio_files(
    audio_files: List[Path],
    model: nn.Module,
    id2artist: Dict[int, str],
    device: str,
    max_frames: int = 3750,
) -> tuple:

    model.eval()
    
    top1_predictions = []
    top3_predictions = {}
    
    with torch.no_grad():
        for idx, audio_file in enumerate(tqdm(audio_files, desc="Predicting")):
            try:
                # Preprocess audio
                input_tensor = preprocess_audio(str(audio_file), max_frames=max_frames)
                input_tensor = input_tensor.to(device)
                
                # output
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                
                top1_idx = np.argmax(probs)
                top1_artist = id2artist[top1_idx]
                top1_predictions.append(top1_artist)
                
                top3_indices = np.argsort(probs)[-3:][::-1]  
                top3_artists = [id2artist[i] for i in top3_indices]
                top3_predictions[f"{idx+1:03d}"] = top3_artists
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                default_artist = id2artist[0]
                top1_predictions.append(default_artist)
                top3_predictions[f"{idx+1:03d}"] = [default_artist] * 3
    
    return top1_predictions, top3_predictions


def main():
    parser = argparse.ArgumentParser("Test Singer Classification Model")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--id2artist_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames", type=int, default=3750,)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load id2artist mapping
    print(f"Loading id2artist mapping from {args.id2artist_json}")
    with open(args.id2artist_json, 'r') as f:
        id2artist_str = json.load(f)
        id2artist = {int(k): v for k, v in id2artist_str.items()}
    
    num_classes = len(id2artist)
    print(f"Number of classes: {num_classes}")
    
    # Build model
    print("Building model...")
    model = CRNN_Model(num_classes=num_classes)
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded successfully (Top-1 Acc: {checkpoint.get('top1_acc', 'N/A')}%)")
    
    # Get audio 
    test_dir = Path(args.test_dir)
    audio_files = sorted(test_dir.glob("*.mp3"))
    
    if not audio_files:
        print(f"No .mp3 files found in {test_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to test")
    
    # Predict
    print("Starting prediction...")
    top1_predictions, top3_predictions = predict_audio_files(
        audio_files, model, id2artist, args.device, args.max_frames
    )
    
    # Save top-1 predictions
    top1_output = output_dir / "top1_predictions_2.json"
    with open(top1_output, 'w') as f:
        json.dump(top1_predictions, f, indent=1)
    print(f"\nTop-1 predictions saved to {top1_output}")
    
    # Save top-3 predictions
    top3_output = output_dir / "top3_predictions_2.json"
    with open(top3_output, 'w') as f:
        json.dump(top3_predictions, f, indent=1)
    print(f"Top-3 predictions saved to {top3_output}")
    
    
    print("\nDone!")

if __name__ == "__main__":
    main()