# Artist20 歌手分類：Log-Mel 前處理 + CRNN 訓練

---

## 目錄結構（預設路徑）

```
artist20/
│  ├─ train_val/                 # 原始音檔: <artist>/<album>/*.mp3
│  │   └─ <artist>/<album>/*.mp3
│  ├─ train.json                 # 訓練清單（相對於 dataset_root 的路徑字串）
│  └─ val.json                   # 驗證清單（相對於 dataset_root 的路徑字串）
├─ song_total_data/              # 前處理後的特徵檔 .pkl/.npz（自動建立）
├─ results/                 
└─ checkpoint/             # 訓練輸出：best_task2.pt, cm_*.png, metrics_*.json, loss_curve.png
```
---

## 環境需求

- Python 3.9+（建議 3.10）
- PyTorch（自動偵測 CUDA / MPS / CPU）
- 套件：`librosa`, `numpy`, `matplotlib`, `tqdm`, `scikit-learn`, `dill`（可選 `torchaudio`）

安裝環境：
```bash
pip install -r requirement.txt
```
## 進行前處理

需將檔案中的路徑換成本機路徑

```bash
python3 process.py
```

成功後，`song_total_data/` 會出現大量特徵檔，每個含：
```python
{
  "logmel": np.ndarray (n_mels, T),  # T 對所有檔案一致（例如 3751）
  "artist": "...", "album": "...", "song": "...",
  "sr": 16000, "n_mels": 128, "n_fft": 2048, "hop_length": 512,
  "max_seconds": 120, "crop_mode": "center"
}
```

## 訓練（CRNN + GRU）

### 啟動指令
```bash
python3 train.py --dataset_root ${artist20_path}  --lr 0.0003     --weight_decay 5e-4     --batch_size 16     --epochs 90     --augment     --label_smoothing 0.15     --mixup_alpha 0     --warmup_epochs 5     --seed 42
```

## 測試

### 生成 id2label 檔案
```bash
python3 id2label.py --dataset_root ${artist20_path} 
```
在 dataset 路徑下生成一個 id2label.json 

### 測試
```bash
python3 test.py --test_dir ${test_path}  --checkpoint  ${checkpoint_path} --id2artist_json ${id2label_path}

```