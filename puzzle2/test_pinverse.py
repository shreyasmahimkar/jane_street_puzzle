import torch
import os
pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    if sd["weight"].shape == (96, 48): inps.append((fname, sd))
    elif sd["weight"].shape == (48, 96): outs.append((fname, sd))

matches = 0
for iname, isd in inps:
    pinv = torch.linalg.pinv(isd["weight"])
    for oname, osd in outs:
        diff = (pinv - osd["weight"]).abs().max().item()
        if diff < 1e-3:
            print(f"Match pinv: {iname} and {oname}")
            matches += 1
print("Total pinv matches:", matches)
