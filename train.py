#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
from pathlib import Path
from typing import List, Tuple
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import dill

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


CHECKPOINT_DIR = Path("./checkpoints")
RESULTS_DIR = Path("./results")
# ================================
class SingerDataset(Dataset):
    def __init__(self, pkl_paths: List[Path], labels: List[int], augment: bool = False, max_frames: int = 3750):
        self.pkl_paths = pkl_paths
        self.labels = labels
        self.augment = augment
        self.max_frames = max_frames
        assert len(pkl_paths) == len(labels)
    
    def __len__(self):
        return len(self.pkl_paths)
    
    def __getitem__(self, idx):
        pkl_path = self.pkl_paths[idx]
        label = self.labels[idx]
        
        # Load preprocessed logmel
        with open(pkl_path, 'rb') as f:
            data = dill.load(f)
        
        logmel = data['logmel']  # (n_mels, T)
        
        # Data augmentation for training
        if self.augment:
            logmel = self._augment(logmel)
        
        # Pad or crop to fixed length
        logmel = self._fix_length(logmel, self.max_frames)
        
        # Normalize
        mean = logmel.mean()
        std = logmel.std() + 1e-8
        logmel = (logmel - mean) / std
        
        # Convert to tensor: (1, n_mels, T) for CNN
        logmel = torch.FloatTensor(logmel).unsqueeze(0)
        
        return logmel, label
    
    #make the length same
    def _fix_length(self, logmel: np.ndarray, target_length: int) -> np.ndarray:
        n_mels, T = logmel.shape
        
        if T > target_length:
            start = (T - target_length) // 2
            logmel = logmel[:, start:start + target_length]
        elif T < target_length:
            pad_width = target_length - T
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            logmel = np.pad(logmel, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        return logmel
    # More aggressive data augmentation
    def _augment(self, logmel: np.ndarray) -> np.ndarray:
        logmel = logmel.copy()
        n_mels, T = logmel.shape
        
        if np.random.rand() < 0.5:
            t_mask = np.random.randint(10, min(50, T//8))
            t_start = np.random.randint(0, max(1, T - t_mask))
            logmel[:, t_start:t_start+t_mask] = logmel.mean()
      
            f_mask = np.random.randint(5, min(20, n_mels//6))
            f_start = np.random.randint(0, max(1, n_mels - f_mask))
            logmel[f_start:f_start+f_mask, :] = logmel.mean()
        
        if np.random.rand() < 0.3:
            crop_size = int(T * np.random.uniform(0.85, 1.0))
            start = np.random.randint(0, max(1, T - crop_size))
            logmel = logmel[:, start:start+crop_size]
            # Pad back to original size
            if crop_size < T:
                pad = T - crop_size
                logmel = np.pad(logmel, ((0, 0), (0, pad)), mode='edge')
        
        # Random noise
        if np.random.rand() < 0.3:
            noise = np.random.randn(*logmel.shape) * 0.05
            logmel = logmel + noise
        
        return logmel

# ================================


class CRNN_Model(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 128, hidden_size: int = 128):
        super(CRNN_Model, self).__init__()
        
        # CNN for feature extraction (with more dropout)
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
        
        # RNN 
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

def pkl_name_for(artist: str, album: str, song: str) -> str:
    return f"{artist}__{album}__{song}.pkl"

def read_file(json_path: Path, pkl_dir: Path) -> List[Tuple[str, str, str, Path]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    items = []
    for i in data:
        i = i.lstrip("./")
        parts = Path(i).parts
        # ('train_val', 'artist', 'album', 'song.mp3')
        artist, album, song = parts[1], parts[2], parts[3]
        pkl_path = pkl_dir / pkl_name_for(artist, album, song)
        items.append((artist, album, song, pkl_path))
    return items

def json_to_pkl_paths(items: List[Tuple[str, str, str, Path]]) -> List[Path]:
    paths = []
    missing_count = 0
    for artist, album, song, p in items:
        if not p.exists():
            if missing_count < 5:  # Only show first 5 warnings
                print(f"[WARN] missing pkl: {p.name}")
            missing_count += 1
        else:
            paths.append(p)
    if missing_count > 0:
        print(f"[WARN] Total missing files: {missing_count}")
    return paths

def peek_artists(pkl_list: List[Path]) -> List[str]:
    labs = []
    for p in pkl_list:
        with open(p, "rb") as fp:
            d = dill.load(fp)
            labs.append(d["artist"])
    return labs

def load_data(dataset_root: Path):
    train_json = dataset_root / "artist20" / "train.json"
    val_json = dataset_root / "artist20" / "val.json"
    pkl_dir = dataset_root / "song_total_data"
    
    print(f"Loading data from {dataset_root}")
    print(f"Train JSON: {train_json}")
    print(f"Val JSON: {val_json}")
    print(f"PKL Directory: {pkl_dir}")
    
    # Read JSON files
    train_items = read_file(train_json, pkl_dir)
    val_items = read_file(val_json, pkl_dir)
    print(f"Num train/val entries: {len(train_items)}, {len(val_items)}")
    
    # Get pkl paths
    train_pkl = json_to_pkl_paths(train_items)
    val_pkl = json_to_pkl_paths(val_items)
    print(f"Valid pkl counts: {len(train_pkl)}, {len(val_pkl)}")
    
    # Get artist labels
    train_labels = peek_artists(train_pkl)
    val_labels = peek_artists(val_pkl)
    
    # Build label encoder
    le = LabelEncoder().fit(train_labels)
    classes = list(le.classes_)
    cls2id = {c: i for i, c in enumerate(classes)}
    id2cls = {i: c for c, i in cls2id.items()}
    
    print(f"Number of classes: {len(classes)}")
    print(f"Train class distribution: {Counter(train_labels).most_common(5)}")
    
    # Encode labels
    train_label_ids = le.transform(train_labels).tolist()
    val_label_ids = le.transform(val_labels).tolist()
    
    return train_pkl, val_pkl, train_label_ids, val_label_ids, id2cls

def sanitize(name: str) -> str:
    import re
    return re.sub(r'[\\/:"*?<>|]+', "_", name)

# ================================

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None, mixup_alpha=0.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Mixup augmentation
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = inputs.size(0)
            index = torch.randperm(batch_size).to(device)
            
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            labels_a, labels_b = labels, labels[index]
            
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # prevent exploding gradients
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

# ================================

def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu())
            all_preds.append(outputs.argmax(1).cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    top1_acc = accuracy_score(all_labels, all_preds) * 100
   
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3_correct = sum([label in top3_preds[i] for i, label in enumerate(all_labels)])
    top3_acc = 100. * top3_correct / len(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return top1_acc, top3_acc, cm, all_preds, all_labels

def plot_confusion_matrix(cm, idx_to_artist, save_path):
    plt.figure(figsize=(16, 14))
    
    artist_names = [idx_to_artist[i] for i in range(len(idx_to_artist))]
    
    # Create heatmap using imshow
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    
    # Set ticks and labels
    tick_marks = np.arange(len(artist_names))
    plt.xticks(tick_marks, artist_names, rotation=90, ha='right')
    plt.yticks(tick_marks, artist_names, rotation=0)
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Validation Set', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# ================================

def main():
    parser = argparse.ArgumentParser("Singer Classification with CNN/CRNN")
    parser.add_argument("--dataset_root", type=str, default="/tmp2/mir/data/hw1", 
                        help="Root directory of dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max_frames", type=int, default=3750)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0)
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create directories
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    dataset_root = Path(args.dataset_root)
    train_pkl, val_pkl, train_labels, val_labels, idx_to_artist = load_data(dataset_root)
    num_classes = len(idx_to_artist)
    
    train_dataset = SingerDataset(train_pkl, train_labels, augment=args.augment, max_frames=args.max_frames)
    val_dataset = SingerDataset(val_pkl, val_labels, augment=False, max_frames=args.max_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Build model
    model = CRNN_Model(num_classes=num_classes)
    print(f"Building model...")
    
    
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use label smoothing to prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warm-up + ReduceLROnPlateau scheduler (better for this case)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True, min_lr=1e-7
    )
    
    # Training
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 15
    train_losses = []
    val_accs = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, 
                                           args.device, grad_clip=args.grad_clip, mixup_alpha=args.mixup_alpha)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        top1_acc, top3_acc, cm, _, _ = evaluate(model, val_loader, args.device, num_classes)
        print(f"Val Top-1 Acc: {top1_acc:.2f}%, Val Top-3 Acc: {top3_acc:.2f}%")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track metrics
        train_losses.append(train_loss)
        val_accs.append(top1_acc)
        
        # Step scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            reduce_scheduler.step(top1_acc)  
        
        # Save best model
        if top1_acc > best_val_acc:
            best_val_acc = top1_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'top1_acc': top1_acc,
                'top3_acc': top3_acc,
            }, CHECKPOINT_DIR / f"crnn_best.pth")
            print(f"✓ Best model saved! Top-1: {top1_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_val_acc:.2f}%)")
        
        # overfitting prevention
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"crnn_training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved to {RESULTS_DIR / f'crnn_training_curves.png'}")
    
    
    print("\n" + "="*50)
    print("Final Evaluation on Best Model")
    print("="*50)
    
    checkpoint = torch.load(CHECKPOINT_DIR / f"crnn_best.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    top1_acc, top3_acc, cm, preds, labels = evaluate(model, val_loader, args.device, num_classes)
    
    print(f"\nValidation Set Results:")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    
    # Save confusion matrix
    cm_path = RESULTS_DIR / f"crnn_confusion_matrix.png"
    plot_confusion_matrix(cm, idx_to_artist, cm_path)
    
    # Save results 
    results_txt = RESULTS_DIR / f"crnn_results.txt"
    with open(results_txt, 'w') as f:
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.2f}%\n")
        f.write(f"\nPer-class accuracy:\n")
        for i in range(num_classes):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == i).sum() / class_mask.sum() * 100
                f.write(f"  {idx_to_artist[i]}: {class_acc:.2f}%\n")
    
    print(f"\nResults saved to {results_txt}")

if __name__ == "__main__":
    main()