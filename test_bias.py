import torch
import os
pieces_dir = "historical_data_and_pieces/pieces"
for fname in sorted(os.listdir(pieces_dir))[:5]:
    sd = torch.load(os.path.join(pieces_dir, fname), map_location="cpu")
    print(fname, sd["bias"][:5])
