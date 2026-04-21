import pickle
import torch
# load piece 0 manually
with open("historical_data_and_pieces/pieces/piece_0.pth", "rb") as f:
    try:
        data = pickle.load(f)
        print("Pickle data:", data)
    except Exception as e:
        print("Error:", e)

