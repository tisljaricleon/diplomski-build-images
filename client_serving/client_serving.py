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
from jtop import jtop
import asyncio
import time

# Jtop stats monitoring & logging
latest_stats = {}
def monitor_jtop():
    try:
        with jtop() as jetson:
            while jetson.ok():
                latest_stats.update(jetson.stats)
    except Exception as e:
        print(f"[JTOP MONITOR] Error: {e}")


def log_resource_usage():
    stat_fields = [
        'timestamp', 'cpu1', 'cpu2', 'cpu3', 'cpu4', 'cpu5', 'cpu6', 'gpu', 'ram', 'swap',
        'fan', 'temp_cpu', 'temp_gpu', 'temp_soc0', 'temp_soc1', 'temp_soc2', 'temp_therm_junction',
        'power_vdd_cpu_gpu_cv', 'power_vdd_soc', 'power_tot', 'jetson_clocks', 'nvp_model'
    ]
    log_path = "/home/model/resource_log.csv"

    if os.path.exists(log_path):
        with open(log_path, 'w', newline='') as file:
            pass

    while True:
        row = [
            datetime.datetime.now().isoformat(),
            latest_stats.get('CPU1', ''),
            latest_stats.get('CPU2', ''),
            latest_stats.get('CPU3', ''),
            latest_stats.get('CPU4', ''),
            latest_stats.get('CPU5', ''),
            latest_stats.get('CPU6', ''),
            latest_stats.get('GPU', ''),
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
        file_exists = os.path.exists(log_path)
        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(stat_fields)
            writer.writerow(row)
        time.sleep(0.5)

threading.Thread(target=monitor_jtop, daemon=True).start()
threading.Thread(target=log_resource_usage, daemon=True).start()


# Defining device, transforms, Net
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


# Model loading
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


# Prediction endpoint
def inference(tensor):
    with torch.no_grad():
        output = model(tensor)
        return output.argmax(dim=1).item()


app = FastAPI()          
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    try:
        if model is None:
            model = load_model()
            if model is None:
                return JSONResponse({"label": None, "error": "Model not found"}, status_code=404)
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = cifar10_transform(image).unsqueeze(0)
        tensor = tensor.to(device)
        prediction = await asyncio.to_thread(inference, tensor)
        return JSONResponse({"label": int(prediction)})
    except Exception as e:
        return JSONResponse({ "label": None, "error": str(e)}, status_code=500)
