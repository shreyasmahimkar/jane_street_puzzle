import os
import torch
import re

pieces_dir = "historical_data_and_pieces/pieces"
files = []
for fname in os.listdir(pieces_dir):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    inode = os.stat(path).st_ino
    num = int(re.findall(r"\d+", fname)[0])
    files.append((inode, num, path))

files.sort() # sort by inode
shapes = []
indices = []
for inode, num, path in files:
    sd = torch.load(path, map_location="cpu")
    if sd["weight"].shape == (96, 48): shapes.append("inp")
    elif sd["weight"].shape == (48, 96): shapes.append("out")
    else: shapes.append("last")
    indices.append(num)

print("Shapes in inode order:")
print(shapes[:10])
print(shapes[48:58])
print("Indices in inode order:")
print(",".join(map(str, indices)))
