#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import dill
from sklearn.preprocessing import LabelEncoder

def pkl_name_for(artist: str, album: str, song: str) -> str:
    return f"{artist}__{album}__{song}.pkl"

def read_file(json_path: Path, pkl_dir: Path) -> List[Tuple[str, str, str, Path]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    items = []
    for i in data:
        i = i.lstrip("./")
        parts = Path(i).parts
        artist, album, song = parts[1], parts[2], parts[3]
        pkl_path = pkl_dir / pkl_name_for(artist, album, song)
        items.append((artist, album, song, pkl_path))
    return items

def json_to_pkl_paths(items: List[Tuple[str, str, str, Path]]) -> List[Path]:
    paths = []
    for artist, album, song, p in items:
        if p.exists():
            paths.append(p)
    return paths

def peek_artists(pkl_list: List[Path]) -> List[str]:
    labs = []
    for p in pkl_list:
        with open(p, "rb") as fp:
            d = dill.load(fp)
            labs.append(d["artist"])
    return labs

def main():
    parser = argparse.ArgumentParser("Save id2artist mapping for testing")
    parser.add_argument("--dataset_root", type=str, default="/tmp2/mir/data/hw1")
    parser.add_argument("--output", type=str, default="./id2label.json")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    train_json = dataset_root / "artist20" / "train.json"
    pkl_dir = dataset_root / "song_total_data"
    
    print(f"Loading training data from {dataset_root}")
    
    # Read training JSON
    train_items = read_file(train_json, pkl_dir)
    train_pkl = json_to_pkl_paths(train_items)
    
    print(f"Found {len(train_pkl)} training files")
    
    # Get artist labels
    train_labels = peek_artists(train_pkl)
    
    # Build label encoder
    le = LabelEncoder().fit(train_labels)
    classes = list(le.classes_)
    
    # Create id2artist dict
    id2label = {i: artist for i, artist in enumerate(classes)}
    
    print(f"Number of artists: {len(id2label)}")
    print(f"Artists: {classes}")
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(id2label, f, indent=2)
    
    print(f"\nid2artist mapping saved to {output_path}")
    print("You can now use this file for testing with --id2artist_json argument")

if __name__ == "__main__":
    main()