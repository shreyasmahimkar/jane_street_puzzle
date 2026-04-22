import torch
import os
import pandas as pd
import re

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    num = int(re.findall(r"\d+", fname)[0])
    if sd["weight"].shape == (96, 48): inps.append((num, sd))
    elif sd["weight"].shape == (48, 96): outs.append((num, sd))

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
x0 = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values[0], dtype=torch.float32)

best_pairs = []
for inum, isd in inps:
    for onum, osd in outs:
        h = torch.nn.functional.linear(x0, isd["weight"], isd["bias"])
        h = torch.nn.functional.relu(h)
        out = torch.nn.functional.linear(h, osd["weight"], osd["bias"])
        
        # Check properties of out
        # e.g., is it close to 0? Is it close to integers?
        best_pairs.append((out.abs().sum().item(), inum, onum, out))

best_pairs.sort()
print("Pairs with smallest absolute L1 sum over x_0:")
for s, i, j, out in best_pairs[:5]:
    print(i, j, s)
