import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import os
import yaml
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import threading
import csv
import datetime
import psutil


# Use tegrastats for GPU monitoring in Docker (Jetson)
import subprocess
def get_tegrastats_stats():
    try:
        output = subprocess.check_output(['tegrastats', '--interval', '1000', '--count', '1'], stderr=subprocess.STDOUT, text=True)
        for line in output.splitlines():
            if 'GR3D_FREQ' in line and 'CPU' in line and 'RAM' in line:
                ram_used = None
                ram_total = None
                swap_used = None
                swap_total = None
                cpu_cores = []
                gpu = None
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'RAM' and i+1 < len(parts):
                        ram_info = parts[i+1]
                        if '/' in ram_info and 'MB' in ram_info:
                            ram_used, ram_total = ram_info.replace('MB','').split('/')
                            ram_used = int(ram_used)
                            ram_total = int(ram_total)
                    if part == 'SWAP' and i+1 < len(parts):
                        swap_info = parts[i+1]
                        if '/' in swap_info and 'MB' in swap_info:
                            swap_used, swap_total = swap_info.replace('MB','').split('/')
                            swap_used = int(swap_used)
                            swap_total = int(swap_total)
                    if part == 'CPU' and i+1 < len(parts):
                        cpu_info = parts[i+1]
                        if cpu_info.startswith('[') and cpu_info.endswith(']'):
                            cpu_core_strs = cpu_info[1:-1].split(',')
                            for core in cpu_core_strs:
                                if '%@' in core:
                                    cpu_cores.append(float(core.split('%@')[0]))
                                else:
                                    cpu_cores.append(None)
                    if part == 'GR3D_FREQ' and i > 0 and '%' in parts[i-1]:
                        gpu_str = parts[i-1]
                        try:
                            gpu = int(gpu_str.strip('%'))
                        except Exception:
                            gpu = None
                return {
                    'ram_used': ram_used,
                    'ram_total': ram_total,
                    'swap_used': swap_used,
                    'swap_total': swap_total,
                    'cpu_cores': cpu_cores,
                    'gpu': gpu
                }
        return None
    except Exception as e:
        print(f"tegrastats not available or failed: {e}")
        return None
NVML_AVAILABLE = True


ongoing_requests = 0
ongoing_requests_lock = threading.Lock()
def log_resource_usage(request_id=None, ongoing=None):
    process = psutil.Process(os.getpid())
    try:
        cpu = process.cpu_percent(interval=0.01)
        print(f"[RESOURCE LOG] CPU usage found: {cpu}%")
    except Exception as e:
        cpu = None
        print(f"[RESOURCE LOG] CPU usage not found: {e}")
    try:
        mem = process.memory_info().rss / (1024 * 1024)  # MB
        print(f"[RESOURCE LOG] Memory usage found: {mem} MB")
    except Exception as e:
        mem = None
        print(f"[RESOURCE LOG] Memory usage not found: {e}")
    ram = None
    ram_total = None
    swap = None
    swap_total = None
    cpu_cores = None
    gpu = None
    if NVML_AVAILABLE:
        try:
            stats = get_tegrastats_stats()
            if stats:
                ram = stats['ram_used']
                ram_total = stats['ram_total']
                swap = stats['swap_used']
                swap_total = stats['swap_total']
                cpu_cores = stats['cpu_cores']
                gpu = stats['gpu']
                print(f"[RESOURCE LOG] (tegrastats) RAM: {ram}/{ram_total} MB, SWAP: {swap}/{swap_total} MB, CPU cores: {cpu_cores}, GPU: {gpu}%")
        except Exception as e:
            print(f"[RESOURCE LOG] tegrastats parsing failed: {e}")
    log_path = "/home/model/resource_log.csv"
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "timestamp", "request_id", "ram_used_mb", "ram_total_mb", "swap_used_mb", "swap_total_mb", "cpu_cores", "gpu_percent", "ongoing_requests"
            ])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            request_id if request_id is not None else '',
            ram,
            ram_total,
            swap,
            swap_total,
            cpu_cores,
            gpu if gpu is not None else '',
            ongoing if ongoing is not None else ''
        ])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

app = FastAPI()

def get_model_path():
    return f"/home/model/model.pt"

def load_model():
    model_path = get_model_path()
    if not os.path.exists(model_path):
        return None
    try:
        print(f"[MODEL LOAD] Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"[MODEL LOAD] Loaded object type: {type(state_dict)}")
        model = Net()
        model.load_state_dict(state_dict)
        print("[MODEL LOAD] Successfully loaded state_dict into Net")
        model.eval()
        print("[MODEL LOAD] Model set to eval mode")
        return model
    except Exception as e:
        print(f"[MODEL LOAD ERROR] {e}")
        return None

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
model = load_model()
if model is not None:
    model.to(device)
    print(f"[INIT] Model moved to device: {device}")
cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, ongoing_requests
    request_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    with ongoing_requests_lock:
        ongoing_requests += 1
        current_ongoing = ongoing_requests
    try:
        if model is None:
            model = load_model()
            if model is None:
                log_resource_usage(request_id, current_ongoing)
                return JSONResponse({"label": None, "error": "Model not found"}, status_code=200)
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = cifar10_transform(image).unsqueeze(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            # Print device info for debugging
            device = next(model.parameters()).device
            print(f"[INFERENCE] Model is on device: {device}")
            print(f"[INFERENCE] Input tensor is on device: {tensor.device}")
            output = model(tensor)
            pred = output.argmax(dim=1).item()
        log_resource_usage(request_id, current_ongoing)
        return JSONResponse({"label": int(pred)})
    except Exception as e:
        log_resource_usage(request_id, current_ongoing)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        with ongoing_requests_lock:
            ongoing_requests -= 1


if __name__ == "__main__":
    with open("client_serving_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    address = config.get("server", {}).get("address", "0.0.0.0:8000")

    if ":" in address:
        host, port_str = address.rsplit(":", 1)
        port = int(port_str)

    uvicorn.run(app, host=host, port=port)
    print(f"Client Server Serving started at {host}:{port}")
