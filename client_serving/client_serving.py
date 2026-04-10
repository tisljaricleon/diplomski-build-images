import traceback
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
from jtop import jtop
import asyncio
import time
from typing import List
import datetime


last_request_gpu_mem = 0
last_request_gpu_mem_max = 0

latest_stats = {}
def monitor_jtop():
    try:
        with jtop() as jetson:
            while jetson.ok():
                latest_stats.update(jetson.stats)
    except Exception as e:
        print(f"[JTOP MONITOR] Error: {e}")



def log_resource_usage():
    global last_request_gpu_mem, last_request_gpu_mem_max
    stat_fields = [
        'timestamp', 'cpu1', 'cpu2', 'cpu3', 'cpu4', 'cpu5', 'cpu6', 'gpu',
        'gpu_allocated_mem', 'gpu_request_mem', 'gpu_request_mem_max', 'gpu_total_mem',
        'ram', 'swap',
        'fan', 'temp_cpu', 'temp_gpu', 'temp_soc0', 'temp_soc1', 'temp_soc2', 'temp_therm_junction',
        'power_vdd_cpu_gpu_cv', 'power_vdd_soc', 'power_tot', 'jetson_clocks', 'nvp_model',
    ]
    log_path = "/home/model/resource_log.csv"

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(stat_fields)

    while True:
        gpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else ''
        gpu_total = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else ''
        row = [
            datetime.datetime.now().isoformat(),
            latest_stats.get('CPU1', ''),
            latest_stats.get('CPU2', ''),
            latest_stats.get('CPU3', ''),
            latest_stats.get('CPU4', ''),
            latest_stats.get('CPU5', ''),
            latest_stats.get('CPU6', ''),
            latest_stats.get('GPU', ''),
            gpu_mem,
            last_request_gpu_mem,
            last_request_gpu_mem_max,
            gpu_total,
            latest_stats.get('RAM', ''),
            latest_stats.get('SWAP', ''),
            latest_stats.get('Fan pwmfan0', ''),
            latest_stats.get('Temp cpu', ''),
            latest_stats.get('Temp gpu', ''),
            latest_stats.get('Temp soc0', ''),
            latest_stats.get('Temp soc1', ''),
            latest_stats.get('Temp soc2', ''),
            latest_stats.get('Temp tj', ''),
            latest_stats.get('Power VDD_CPU_GPU_CV', ''),
            latest_stats.get('Power VDD_SOC', ''),
            latest_stats.get('Power TOT', ''),
            latest_stats.get('jetson_clocks', ''),
            latest_stats.get('nvp model', ''),
        ]
        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        time.sleep(0.25)

threading.Thread(target=monitor_jtop, daemon=True).start()
threading.Thread(target=log_resource_usage, daemon=True).start()



cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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


def load_model():
    model_path = "/home/model/model.pt"
    if not os.path.exists(model_path):
        return None
    try:
        print(f"[MODEL LOAD] Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"[MODEL LOAD] Loaded object type: {type(state_dict)}")
        model = Net()
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        print(f"[MODEL LOAD] Model moved to device: {device}")
        return model
    except Exception as e:
        print(f"[MODEL LOAD] Error: {e}")
        return None


model = load_model()

LABEL_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def inference(tensor):
    with torch.no_grad():
        output = model(tensor)
        preds = output.argmax(dim=1)
        return preds.cpu().numpy(), output.cpu().numpy()


app = FastAPI()

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    global model, last_request_gpu_mem, last_request_gpu_mem_max
    try:
        if model is None:
            model = load_model()
        if model is None:
            print("[PREDICT] Model not found")
            return JSONResponse({"results": None, "error": "Model not found"}, status_code=404)

        images = []
        for file in files:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
            tensor = cifar10_transform(image)
            images.append(tensor)
        batch_tensor = torch.stack(images).to(device)


        mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        mem_max_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        start_time = time.time()
        print(f"[PREDICT] Inference start: {datetime.datetime.now().isoformat()}")
        preds, logits = await asyncio.to_thread(inference, batch_tensor)
        end_time = time.time()
        print(f"[PREDICT] Inference end: {datetime.datetime.now().isoformat()}, duration: {end_time - start_time:.4f} seconds")

        mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        mem_max_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        last_request_gpu_mem = mem_after - mem_before
        last_request_gpu_mem_max = mem_max_after - mem_max_before

        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1).numpy()
        results = []
        for idx in range(len(preds)):
            label_idx = int(preds[idx])
            label_name = LABEL_NAMES[label_idx]
            confidence = float(probs[idx][label_idx])
            results.append({
                "label_index": label_idx,
                "label_name": label_name,
                "confidence": confidence
            })
        return JSONResponse({"results": results}, status_code=200)
    except Exception as e:
        print(f"[PREDICT] {e}")
        traceback.print_exc()
        return JSONResponse({ "results": None, "error": str(e)}, status_code=500)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"[GLOBAL EXCEPTION] {exc}")
    traceback.print_exc()
    return JSONResponse({"results": None, "error": str(exc)}, status_code=500)
