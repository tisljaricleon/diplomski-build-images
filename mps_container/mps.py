import torch

def main():
    import time
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    # Long-running workload (~5 minutes)
    x = torch.randn(10000, 10000, device='cuda')
    y = torch.randn(10000, 10000, device='cuda')
    start = time.time()
    duration = 300  # 5 minutes
    count = 0
    print("Starting long computation loop for ~5 minutes...")
    while time.time() - start < duration:
        z = torch.matmul(x, y)
        x = z
        count += 1
        if count % 10 == 0:
            print(f"Iteration {count}, elapsed: {int(time.time() - start)}s")
    print(f"Done. Ran {count} iterations. Final result shape: {z.shape}")

if __name__ == "__main__":
    main()
