import torch
import os

pieces_dir = "historical_data_and_pieces/pieces"
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    print(f"{fname}:")
    for k, v in sd.items():
        print(f"  {k}: {v.shape}")

