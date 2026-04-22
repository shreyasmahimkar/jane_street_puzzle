import torch
import os
import pandas as pd
import re
from scipy.optimize import linear_sum_assignment

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in os.listdir(pieces_dir):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    num = int(re.findall(r"\d+", fname)[0])
    if sd["weight"].shape == (96, 48): inps.append((num, sd))
    elif sd["weight"].shape == (48, 96): outs.append((num, sd))

inps.sort(key=lambda x: x[0])
outs.sort(key=lambda x: x[0])
last_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

# 1. Pairing via Hungarian Algorithm on negative diagonal structure
# We want to MINIMIZE the diagonal sum (since it's strong negative)
cost_matrix = torch.zeros(48, 48)
for i, (inum, isd) in enumerate(inps):
    for j, (onum, osd) in enumerate(outs):
        prod = torch.matmul(osd["weight"], isd["weight"]) # 48x96 @ 96x48 = 48x48
        diag_sum = torch.trace(prod).item()
        cost_matrix[i, j] = diag_sum

# linear_sum_assignment solves min cost
row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

blocks = []
for idx in range(48):
    i = row_ind[idx]
    j = col_ind[idx]
    inum = inps[i][0]
    onum = outs[j][0]
    w_out_norm = outs[j][1]["weight"].norm(p='fro').item()
    blocks.append({'inp': inps[i][1], 'out': outs[j][1], 'inum': inum, 'onum': onum, 'norm': w_out_norm})

# 2. Ordering: Seed by W_out frobenius norm ascending or descending?
blocks.sort(key=lambda b: b['norm']) # Try ascending

# 3. Bubble repair refinement
df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)

def evaluate(blocks_seq):
    curr = X
    for b in blocks_seq:
        h = torch.nn.functional.linear(curr, b['inp']["weight"], b['inp']["bias"])
        h = torch.nn.functional.relu(h)
        curr = curr + torch.nn.functional.linear(h, b['out']["weight"], b['out']["bias"])
    preds = torch.nn.functional.linear(curr, last_sd["weight"], last_sd["bias"])
    return torch.nn.functional.mse_loss(preds, Y).item()

best_loss = evaluate(blocks)
print(f"Initial matched & norm-sorted loss: {best_loss:.6f}")

import time
t0 = time.time()
changed = True
while changed:
    changed = False
    for i in range(47):
        new_blocks = list(blocks)
        new_blocks[i], new_blocks[i+1] = new_blocks[i+1], new_blocks[i]
        l = evaluate(new_blocks)
        if l < best_loss:
            best_loss = l
            blocks = new_blocks
            changed = True
            print(f"Swapped {i} and {i+1}, new loss: {best_loss:.6f}")

print(f"Final loss after bubble repair: {best_loss:.6f}")

ans = []
for b in blocks:
    ans.append(b['inum'])
    ans.append(b['onum'])
ans.append(85)
print("SOLUTION:")
print(",".join(map(str, ans)))

