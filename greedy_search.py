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

curr_X = X
avail_inps = list(inps)
avail_outs = list(outs)

sequence = []

for step in range(48):
    best_loss = float('inf')
    best_pair = None
    best_X = None
    best_idx = None
    
    for i, (inum, isd) in enumerate(avail_inps):
        for j, (onum, osd) in enumerate(avail_outs):
            h = torch.nn.functional.linear(curr_X, isd["weight"], isd["bias"])
            h = torch.nn.functional.relu(h)
            next_X = curr_X + torch.nn.functional.linear(h, osd["weight"], osd["bias"])
            
            preds = torch.nn.functional.linear(next_X, last_sd["weight"], last_sd["bias"])
            loss = torch.nn.functional.mse_loss(preds, Y).item()
            
            if loss < best_loss:
                best_loss = loss
                best_pair = (inum, onum)
                best_X = next_X
                best_idx = (i, j)
                
    curr_X = best_X
    sequence.append(best_pair)
    avail_inps.pop(best_idx[0])
    avail_outs.pop(best_idx[1])
    print(f"Step {step+1:02d}: inp {best_pair[0]:02d}, out {best_pair[1]:02d} | loss = {best_loss:.6f}")

preds = torch.nn.functional.linear(curr_X, last_sd["weight"], last_sd["bias"])
final_loss = torch.nn.functional.mse_loss(preds, Y).item()
print("Final loss:", final_loss)

ans = []
for inum, onum in sequence:
    ans.append(inum)
    ans.append(onum)
ans.append(85)
print("SOLUTION:")
print(",".join(map(str, ans)))

