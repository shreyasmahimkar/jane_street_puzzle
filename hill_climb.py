import torch
import os
import pandas as pd
import re
import random
import time

pieces_dir = "historical_data_and_pieces/pieces"
inps_dict, outs_dict = {}, {}
for fname in os.listdir(pieces_dir):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    num = int(re.findall(r"\d+", fname)[0])
    if sd["weight"].shape == (96, 48): inps_dict[num] = sd
    elif sd["weight"].shape == (48, 96): outs_dict[num] = sd

last_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)

def evaluate(seq, x_data, y_data):
    curr = x_data
    for inum, onum in seq:
        isd = inps_dict[inum]
        osd = outs_dict[onum]
        h = torch.nn.functional.linear(curr, isd["weight"], isd["bias"])
        h = torch.nn.functional.relu(h)
        curr = curr + torch.nn.functional.linear(h, osd["weight"], osd["bias"])
    preds = torch.nn.functional.linear(curr, last_sd["weight"], last_sd["bias"])
    return torch.nn.functional.mse_loss(preds, y_data).item()

# Start with beam search result!
start_seq_flat = [87,71,31,36,58,78,91,51,73,72,41,75,86,26,68,93,10,55,49,6,69,20,0,54,42,33,13,89,16,38,48,22,4,53,62,12,1,76,14,47,27,8,95,7,84,9,2,32,61,25,88,92,3,40,56,96,45,46,39,34,23,57,44,90,64,83,28,19,18,67,77,70,35,52,74,80,94,63,50,21,59,79,5,11,65,82,60,30,15,66,43,17,37,29,81,24]
seq = []
for i in range(0, len(start_seq_flat), 2):
    seq.append((start_seq_flat[i], start_seq_flat[i+1]))

best_loss = evaluate(seq, X, Y)
print("Initial loss:", best_loss)

# hill climbing
t0 = time.time()
stalled = 0
while best_loss > 1e-4 and stalled < 5000:
    # try random swap
    i, j = random.sample(range(48), 2)
    new_seq = list(seq)
    
    # swap options:
    # 1. swap whole blocks
    # 2. swap just inps
    # 3. swap just outs
    action = random.choice([1, 2, 3])
    if action == 1:
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    elif action == 2:
        new_seq[i] = (new_seq[j][0], new_seq[i][1])
        new_seq[j] = (new_seq[i][0], new_seq[j][1])
    else:
        new_seq[i] = (new_seq[i][0], new_seq[j][1])
        new_seq[j] = (new_seq[j][0], new_seq[i][1])
        
    l = evaluate(new_seq, X, Y)
    if l < best_loss: # accept if better
        best_loss = l
        seq = new_seq
        stalled = 0
        print(f"New best loss: {best_loss:.6f} (time: {time.time()-t0:.1f}s)")
    else:
        # maybe simulated annealing
        # with very small probability accept worse
        temp = 0.0001
        import math
        if random.random() < math.exp((best_loss - l)/temp):
            seq = new_seq
        stalled += 1

print("Final loss:", best_loss)
ans = []
for i, o in seq:
    ans.extend([i, o])
ans.append(85)
print("SOLUTION:")
print(",".join(map(str, ans)))

