import torch
import os
import pandas as pd
import re
import time

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in os.listdir(pieces_dir):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    num = int(re.findall(r"\d+", fname)[0])
    if sd["weight"].shape == (96, 48): inps.append((num, sd))
    elif sd["weight"].shape == (48, 96): outs.append((num, sd))

# Sort to maintain predictable order
inps.sort(key=lambda x: x[0])
outs.sort(key=lambda x: x[0])

last_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)

# use 500 samples for speed during search
x0 = X[:500]
y0 = Y[:500]

beam = [(float('inf'), x0, set(), set(), [])]
BEAM_WIDTH = 50

t0 = time.time()
for step in range(48):
    new_candidates = []
    
    for curr_loss, curr_X, used_i, used_o, seq in beam:
        for orig_i, (inum, isd) in enumerate(inps):
            if orig_i in used_i: continue
            # compute h
            h = torch.nn.functional.linear(curr_X, isd["weight"], isd["bias"])
            h = torch.nn.functional.relu(h)
            
            for orig_o, (onum, osd) in enumerate(outs):
                if orig_o in used_o: continue
                
                next_X = curr_X + torch.nn.functional.linear(h, osd["weight"], osd["bias"])
                preds = torch.nn.functional.linear(next_X, last_sd["weight"], last_sd["bias"])
                
                loss = torch.nn.functional.mse_loss(preds, y0).item()
                
                new_used_i = used_i | {orig_i}
                new_used_o = used_o | {orig_o}
                new_candidates.append((loss, next_X, new_used_i, new_used_o, seq + [(inum, onum)]))
                
    new_candidates.sort(key=lambda x: x[0])
    # to avoid identical selections with slightly different paths (though sets should prevent permutations of same layers? Wait, sequence matters)
    # just keep top BEAM_WIDTH
    beam = new_candidates[:BEAM_WIDTH]
    
    print(f"Step {step+1}, best loss: {beam[0][0]:.6f}, time: {time.time()-t0:.1f}s")

best_loss, best_X, _, _, best_seq = beam[0]
print("Final loss subset:", best_loss)

# evaluate on full X
curr_X = X
for inum, onum in best_seq:
    isd = next(x[1] for x in inps if x[0] == inum)
    osd = next(x[1] for x in outs if x[0] == onum)
    curr_X = curr_X + torch.nn.functional.linear(
        torch.nn.functional.relu(torch.nn.functional.linear(curr_X, isd["weight"], isd["bias"])),
        osd["weight"], osd["bias"]
    )
full_preds = torch.nn.functional.linear(curr_X, last_sd["weight"], last_sd["bias"])
print("Final absolute full loss:", torch.nn.functional.mse_loss(full_preds, Y).item())

ans = []
for inum, onum in best_seq:
    ans.append(inum)
    ans.append(onum)
print("SOLUTION:")
print(",".join(map(str, ans)))

