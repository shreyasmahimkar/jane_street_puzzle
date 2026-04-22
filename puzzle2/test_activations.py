import torch
import os
import pandas as pd
pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    if sd["weight"].shape == (96, 48): inps.append(sd)
    elif sd["weight"].shape == (48, 96): outs.append(sd)

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)

best_var = float('inf')
results = []
for i, isd in enumerate(inps):
    for j, osd in enumerate(outs):
        x = X[:100] # Use 100 samples
        res = x
        x = torch.nn.functional.linear(x, isd["weight"], isd["bias"])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, osd["weight"], osd["bias"])
        x = res + x
        stats = x.std().item()
        results.append((stats, i, j))

results.sort()
print("Smallest std:")
for s, i, j in results[:5]:
    print(f"std: {s:.4f}, inp: {i}, out: {j}")
    
print("Largest std:")
for s, i, j in results[-5:]:
    print(f"std: {s:.4f}, inp: {i}, out: {j}")
