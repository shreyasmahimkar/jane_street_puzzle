import torch
import sys

print("Loading model...")
model = torch.load('model_3_11.pt', map_location='cpu', weights_only=False)

print("Extracting linears...")
linears = [x for x in model if isinstance(x, torch.nn.Linear)]

print(f"Total linear layers: {len(linears)}")

if len(linears) >= 2:
    second_to_last = linears[-2]
    print(f"Second to last layer bias shape: {second_to_last.bias.shape}")
    bias = second_to_last.bias.detach().cpu().long().numpy()
    print("Bias values:", bias)
    
    n_bytes = 16
    assert len(bias) == n_bytes * 3
    x_minus_v_minus_1 = bias[:n_bytes]
    x_minus_v = bias[n_bytes:2*n_bytes]
    x_minus_v_plus_1 = bias[2*n_bytes:]
    
    print("bias[0:16]: ", bias[:16])
    print("bias[16:32]:", bias[16:32])
    print("bias[32:48]:", bias[32:48])
    
    x = -bias[16:32]
    print("Hash bytes (x):", x)
    
    try:
        hex_hash = bytes(x.tolist()).hex()
        print("MD5 Hash:", hex_hash)
    except Exception as e:
        print("Error converting to hex:", e)

