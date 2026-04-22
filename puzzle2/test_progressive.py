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

last_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)

# we test on the first 1000 items
x0 = X[:1000]
y0 = Y[:1000]

# current naive MSE
base_preds = torch.nn.functional.linear(x0, last_sd["weight"], last_sd["bias"])
print("Base MSE:", torch.nn.functional.mse_loss(base_preds, y0).item())

results = []
for inum, isd in inps:
    for onum, osd in outs:
        h = torch.nn.functional.linear(x0, isd["weight"], isd["bias"])
        h = torch.nn.functional.relu(h)
        x1 = x0 + torch.nn.functional.linear(h, osd["weight"], osd["bias"])
        preds = torch.nn.functional.linear(x1, last_sd["weight"], last_sd["bias"])
        loss = torch.nn.functional.mse_loss(preds, y0).item()
        results.append((loss, inum, onum))

results.sort()
print("Best blocks for step 1:")
for loss, inum, onum in results[:5]:
    print(f"loss: {loss:.4f}, inp: {inum}, out: {onum}")

