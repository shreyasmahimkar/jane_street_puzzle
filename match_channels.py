import torch
import os
import re

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in os.listdir(pieces_dir):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    num = int(re.findall(r"\d+", fname)[0])
    if sd["weight"].shape == (96, 48): inps.append((num, sd))
    elif sd["weight"].shape == (48, 96): outs.append((num, sd))

# For each inp, compute the norm over dim 1 (the 48 in_dim), yielding 96 norms
# For each out, compute the norm over dim 0 (the 48 out_dim), yielding 96 norms
inp_norms = []
for num, sd in inps:
    # weight is (96, 48). norm along dim 1 -> (96,)
    n = sd["weight"].norm(dim=1)
    inp_norms.append((num, n))

out_norms = []
for num, sd in outs:
    # weight is (48, 96). norm along dim 0 -> (96,)
    n = sd["weight"].norm(dim=0)
    out_norms.append((num, n))

matches = []
for i_num, i_n in inp_norms:
    best_dist = float('inf')
    best_o = -1
    for o_num, o_n in out_norms:
        # try sorting the norms to see if they are permutation invariant? No, channels are in order!
        dist = (i_n - o_n).abs().sum().item()
        if dist < best_dist:
            best_dist = dist
            best_o = o_num
    matches.append((best_dist, i_num, best_o))

matches.sort()
for d, i, o in matches[:10]:
    print(f"Dist: {d:.4f}, Inp: {i}, Out: {o}")

