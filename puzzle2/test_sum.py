import torch
import os

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    num = int(fname.split('_')[1].split('.')[0])
    if sd["weight"].shape == (96, 48): inps.append((num, sd["weight"].sum().item()))
    elif sd["weight"].shape == (48, 96): outs.append((num, sd["weight"].sum().item()))

inps.sort(key=lambda x: x[1])
outs.sort(key=lambda x: x[1])

print("Inp sums:", [round(x[1], 4) for x in inps[:10]])
print("Out sums:", [round(x[1], 4) for x in outs[:10]])
