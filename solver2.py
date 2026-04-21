import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import math

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    if sd["weight"].shape == (96, 48): inps.append((fname, sd))
    elif sd["weight"].shape == (48, 96): outs.append((fname, sd))

last_layer_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

class SoftRouter(nn.Module):
    def __init__(self, inps, outs, last_layer_sd):
        super().__init__()
        self.num_blocks = len(inps)
        
        self.inp_weights = nn.Parameter(torch.stack([sd["weight"] for _, sd in inps]), requires_grad=False)
        self.inp_bias = nn.Parameter(torch.stack([sd["bias"] for _, sd in inps]), requires_grad=False)
        self.out_weights = nn.Parameter(torch.stack([sd["weight"] for _, sd in outs]), requires_grad=False)
        self.out_bias = nn.Parameter(torch.stack([sd["bias"] for _, sd in outs]), requires_grad=False)
        self.last_weight = nn.Parameter(last_layer_sd["weight"], requires_grad=False)
        self.last_bias = nn.Parameter(last_layer_sd["bias"], requires_grad=False)
        
        self.A_inp = nn.Parameter(torch.randn(self.num_blocks, self.num_blocks) * 0.01)
        self.A_out = nn.Parameter(torch.randn(self.num_blocks, self.num_blocks) * 0.01)
        
    def forward(self, x, temp=1.0):
        # Sinkhorn-like or just softmax
        P_inp = torch.nn.functional.softmax(self.A_inp / temp, dim=-1)
        P_out = torch.nn.functional.softmax(self.A_out / temp, dim=-1)
        
        cw_inp = torch.einsum('ij, jkl -> ikl', P_inp, self.inp_weights)
        cb_inp = torch.einsum('ij, jk -> ik', P_inp, self.inp_bias)
        cw_out = torch.einsum('ij, jkl -> ikl', P_out, self.out_weights)
        cb_out = torch.einsum('ij, jk -> ik', P_out, self.out_bias)
        
        for i in range(self.num_blocks):
            residual = x
            x = torch.nn.functional.linear(x, cw_inp[i], cb_inp[i])
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.linear(x, cw_out[i], cb_out[i])
            x = residual + x
            
        x = torch.nn.functional.linear(x, self.last_weight, self.last_bias)
        return x

model = SoftRouter(inps, outs, last_layer_sd)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

temp = 1.0
print("Training continuous router...")
for epoch in range(15):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch, temp=temp)
        loss = nn.functional.mse_loss(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    if temp > 0.1:
        temp *= 0.8
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}, Temp: {temp:.3f}")
