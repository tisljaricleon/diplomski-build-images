# test_gpu.py
import torch

device = "cuda"
x = torch.randn(8000, 8000, device=device)

for i in range(50):
    y = torch.mm(x, x)

print("done")