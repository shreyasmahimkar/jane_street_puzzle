import torch
import os
pieces_dir = "historical_data_and_pieces/pieces"
shapes = []
for i in range(97):
    fname = f"piece_{i}.pth"
    sd = torch.load(os.path.join(pieces_dir, fname), map_location="cpu")
    if sd["weight"].shape == (96, 48): shapes.append(0)
    elif sd["weight"].shape == (48, 96): shapes.append(1)
    else: shapes.append(2)
print(shapes)
