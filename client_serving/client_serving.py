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

import asyncio
import time
from typing import List


# 

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
    global model
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

        start_time = time.time()
        preds, logits = await asyncio.to_thread(inference, batch_tensor)
        end_time = time.time()
        print(f"[PREDICT] duration: {end_time - start_time:.4f}s")

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
