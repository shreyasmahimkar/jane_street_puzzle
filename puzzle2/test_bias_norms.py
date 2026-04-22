import torch
import os
import math

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    if sd["weight"].shape == (96, 48): inps.append((fname, sd["bias"].norm().item()))
    elif sd["weight"].shape == (48, 96): outs.append((fname, sd["bias"].norm().item()))

inps.sort(key=lambda x: x[1])
outs.sort(key=lambda x: x[1])
for i in range(5):
    print(f"inp: {inps[i][1]:.6f}, out: {outs[i][1]:.6f}")

