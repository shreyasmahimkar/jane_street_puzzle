import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

pieces_dir = "historical_data_and_pieces/pieces"
inps, outs = [], []
for fname in sorted(os.listdir(pieces_dir)):
    if not fname.endswith(".pth"): continue
    path = os.path.join(pieces_dir, fname)
    sd = torch.load(path, map_location="cpu")
    if sd["weight"].shape == (96, 48): inps.append((fname, sd))
    elif sd["weight"].shape == (48, 96): outs.append((fname, sd))

last_layer_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

print(f"Loaded {len(inps)} inps, {len(outs)} outs")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

class MatchingNetwork(nn.Module):
    def __init__(self, inps, outs, last_layer_sd):
        super().__init__()
        self.num_blocks = len(inps)
        
        # Store all possible weights
        self.inp_weights = nn.Parameter(torch.stack([sd["weight"] for _, sd in inps]), requires_grad=False)
        self.inp_bias = nn.Parameter(torch.stack([sd["bias"] for _, sd in inps]), requires_grad=False)
        
        self.out_weights = nn.Parameter(torch.stack([sd["weight"] for _, sd in outs]), requires_grad=False)
        self.out_bias = nn.Parameter(torch.stack([sd["bias"] for _, sd in outs]), requires_grad=False)
        
        self.last_weight = nn.Parameter(last_layer_sd["weight"], requires_grad=False)
        self.last_bias = nn.Parameter(last_layer_sd["bias"], requires_grad=False)
        
        # Learnable logits for permutations
        # P_inp[i, j] matches block i to inp j
        self.P_inp_logits = nn.Parameter(torch.randn(self.num_blocks, self.num_blocks))
        self.P_out_logits = nn.Parameter(torch.randn(self.num_blocks, self.num_blocks))
        
    def get_permutations(self, temp=1.0, hard=False):
        # We can just use gumbel softmax over the rows, though we really want a doubly stochastic matrix.
        # For simplicity, let's just do softmax over rows for now and see if it converges.
        P_inp = torch.nn.functional.gumbel_softmax(self.P_inp_logits, tau=temp, hard=hard, dim=-1)
        P_out = torch.nn.functional.gumbel_softmax(self.P_out_logits, tau=temp, hard=hard, dim=-1)
        return P_inp, P_out
        
    def forward(self, x, temp=1.0, hard=False):
        P_inp, P_out = self.get_permutations(temp, hard)
        
        # Select active weights
        # expected shape for blocks: (num_blocks, out_features, in_features)
        # block i uses sum_j P_inp[i, j] * inp_weights[j]
        curr_inp_w = torch.einsum('ij, jkl -> ikl', P_inp, self.inp_weights)
        curr_inp_b = torch.einsum('ij, jk -> ik', P_inp, self.inp_bias)
        
        curr_out_w = torch.einsum('ij, jkl -> ikl', P_out, self.out_weights)
        curr_out_b = torch.einsum('ij, jk -> ik', P_out, self.out_bias)
        
        for i in range(self.num_blocks):
            residual = x
            x = torch.nn.functional.linear(x, curr_inp_w[i], curr_inp_b[i])
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.linear(x, curr_out_w[i], curr_out_b[i])
            x = residual + x
            
        x = torch.nn.functional.linear(x, self.last_weight, self.last_bias)
        return x

model = MatchingNetwork(inps, outs, last_layer_sd)
optimizer = torch.optim.Adam([model.P_inp_logits, model.P_out_logits], lr=0.1)

temp = 5.0
for epoch in range(10):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch, temp=temp, hard=False)
        loss = nn.functional.mse_loss(preds, y_batch)
        # Add a penalty to encourage doubly stochastic
        P_inp, P_out = model.get_permutations(temp, hard=False)
        penalty = ((P_inp.sum(0) - 1)**2).mean() + ((P_out.sum(0) - 1)**2).mean()
        (loss + penalty).backward()
        optimizer.step()
        total_loss += loss.item()
    temp = max(0.5, temp * 0.8)
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}, temp: {temp:.2f}")

P_inp_hard, P_out_hard = model.get_permutations(temp=0.1, hard=True)
model.eval()
preds = model(X[:100], temp=0.1, hard=True)
print("Final hard loss:", nn.functional.mse_loss(preds, Y[:100]).item())
