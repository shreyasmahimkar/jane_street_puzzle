import torch
import os
import pandas as pd
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

inps.sort(key=lambda x: x[0])
outs.sort(key=lambda x: x[0])

last_layer_sd = torch.load(os.path.join(pieces_dir, "piece_85.pth"), map_location="cpu")

df = pd.read_csv("historical_data_and_pieces/historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
Y = torch.tensor(df["pred"].values, dtype=torch.float32).unsqueeze(1)

X_batch = X[:500]
Y_batch = Y[:500]

class ParallelRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_blocks = 48
        self.inp_weights = torch.stack([sd["weight"] for _, sd in inps])
        self.inp_bias = torch.stack([sd["bias"] for _, sd in inps])
        self.out_weights = torch.stack([sd["weight"] for _, sd in outs])
        self.out_bias = torch.stack([sd["bias"] for _, sd in outs])
        self.P_logits = torch.nn.Parameter(torch.zeros(48, 48))
        
    def forward(self, x, temp=1.0):
        P = torch.nn.functional.softmax(self.P_logits / temp, dim=-1)
        
        # P[i, j] matches inp i with out j
        cw_out = torch.einsum('ij, jkl -> ikl', P, self.out_weights)
        cb_out = torch.einsum('ij, jk -> ik', P, self.out_bias)
        
        total_residual = x
        for i in range(48):
            h = torch.nn.functional.linear(x, self.inp_weights[i], self.inp_bias[i])
            h = torch.nn.functional.relu(h)
            out_res = torch.nn.functional.linear(h, cw_out[i], cb_out[i])
            total_residual = total_residual + out_res
            
        preds = torch.nn.functional.linear(total_residual, last_layer_sd["weight"], last_layer_sd["bias"])
        return preds

model = ParallelRouter()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

temp = 1.0
import time
t0 = time.time()
for epoch in range(100):
    optimizer.zero_grad()
    preds = model(X_batch, temp=temp)
    loss = torch.nn.functional.mse_loss(preds, Y_batch)
    
    P = torch.nn.functional.softmax(model.P_logits / temp, dim=-1)
    entropy = -torch.mean(torch.sum(P * torch.log(P + 1e-8), dim=-1))
    col_sum = P.sum(0)
    doubly_stoch_loss = torch.mean((col_sum - 1.0)**2)
    
    total_obj = loss + 0.1 * doubly_stoch_loss + temp * 0.01 * entropy
    total_obj.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Temp: {temp:.3f}")
    if loss.item() < 0.05:
        temp *= 0.8

P = torch.nn.functional.softmax(model.P_logits / 0.01, dim=-1)
print("Final loss:", torch.nn.functional.mse_loss(model(X_batch, temp=0.01), Y_batch).item())

