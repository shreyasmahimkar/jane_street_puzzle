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
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)

x0 = X[:1000]

results = []
for num, sd in inps:
    h = torch.nn.functional.linear(x0, sd["weight"], sd["bias"])
    act = torch.nn.functional.relu(h)
    sparsity = (act == 0).float().mean().item()
    results.append((sparsity, num))

results.sort(reverse=True)
for s, n in results[:5]:
    print(f"Sparsity: {s:.4f}, Inp: {n}")
print("---")
for s, n in results[-5:]:
    print(f"Sparsity: {s:.4f}, Inp: {n}")

