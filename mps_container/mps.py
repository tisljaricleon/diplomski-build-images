import torch

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    # Example workload
    x = torch.randn(10000, 10000, device='cuda')
    y = torch.randn(10000, 10000, device='cuda')
    z = torch.matmul(x, y)
    print("Computation done. Result shape:", z.shape)

if __name__ == "__main__":
    main()
