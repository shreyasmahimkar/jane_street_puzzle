import torch
import os
import struct

pieces_dir = "historical_data_and_pieces/pieces"
for fname in sorted(os.listdir(pieces_dir))[:5]:
    sd = torch.load(os.path.join(pieces_dir, fname), map_location="cpu")
    val = sd["bias"][0].item()
    # convert float to binary string
    bits = bin(struct.unpack('!I', struct.pack('!f', val))[0])[2:].zfill(32)
    print(fname, bits)
    
