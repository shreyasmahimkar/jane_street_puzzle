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

inps.sort(key=lambda x: x[0])
outs.sort(key=lambda x: x[0])

last_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)

curr = X
seq = []
for i in range(48):
    inum, isd = inps[i]
    onum, osd = outs[i]
    h = torch.nn.functional.linear(curr, isd["weight"], isd["bias"])
    h = torch.nn.functional.relu(h)
    curr = curr + torch.nn.functional.linear(h, osd["weight"], osd["bias"])
    seq.append(inum)
    seq.append(onum)

preds = torch.nn.functional.linear(curr, last_sd["weight"], last_sd["bias"])
loss = torch.nn.functional.mse_loss(preds, Y).item()

print("Loss if paired in sorted order:", loss)

# What if we pair them as they appear in the os.listdir or something?
# What if it's Block 1 = piece 0 and piece 1 (but wait, they are both inp!)
