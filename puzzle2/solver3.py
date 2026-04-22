import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import re

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    num = int(re.findall(r"\d+", fname)[0])
    if sd["weight"].shape == (96, 48): inps.append((num, sd))
    elif sd["weight"].shape == (48, 96): outs.append((num, sd))

# sort them by name/num just to have a deterministic order
inps.sort(key=lambda x: x[0])
outs.sort(key=lambda x: x[0])

last_layer_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

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
        
        self.A_inp = nn.Parameter(torch.zeros(self.num_blocks, self.num_blocks))
        self.A_out = nn.Parameter(torch.zeros(self.num_blocks, self.num_blocks))
        
    def get_P(self, temp):
        # row-wise softmax
        P_inp = torch.nn.functional.softmax(self.A_inp / temp, dim=-1)
        P_out = torch.nn.functional.softmax(self.A_out / temp, dim=-1)
        return P_inp, P_out
        
    def forward(self, x, temp=1.0):
        P_inp, P_out = self.get_P(temp)
        
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
import time
t0 = time.time()
best_loss = float('inf')

for epoch in range(50):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch, temp=temp)
        loss = nn.functional.mse_loss(preds, y_batch)
        
        # Add penalty to make matrices doubly stochastic and sparse
        P_inp, P_out = model.get_P(temp)
        entropy_inp = -torch.mean(torch.sum(P_inp * torch.log(P_inp + 1e-8), dim=-1))
        entropy_out = -torch.mean(torch.sum(P_out * torch.log(P_out + 1e-8), dim=-1))
        col_sum_inp = P_inp.sum(0)
        col_sum_out = P_out.sum(0)
        doubly_stoch_loss = torch.mean((col_sum_inp - 1.0)**2) + torch.mean((col_sum_out - 1.0)**2)
        
        total_obj = loss + 0.1 * doubly_stoch_loss + temp * 0.1 * (entropy_inp + entropy_out)
        
        total_obj.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Temp: {temp:.3f}")
    if avg_loss < 0.05:
        temp = max(0.01, temp * 0.8)
    if avg_loss < 1e-4:
        break

P_inp, P_out = model.get_P(0.01)
inp_indices = P_inp.argmax(dim=-1).tolist()
out_indices = P_out.argmax(dim=-1).tolist()

# construct final answer
ans = []
for i in range(48):
    ans.append(inps[inp_indices[i]][0])
    ans.append(outs[out_indices[i]][0])
ans.append(85)
print("SOLUTION:")
print(",".join(map(str, ans)))
