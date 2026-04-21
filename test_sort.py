import torch
import os
pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in os.listdir(pieces_dir):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    if sd["weight"].shape == (96, 48): inps.append((fname, sd["weight"][0,0].item()))
    elif sd["weight"].shape == (48, 96): outs.append((fname, sd["weight"][0,0].item()))

print(sorted([x[1] for x in inps])[:5])
print(sorted([x[1] for x in outs])[:5])
