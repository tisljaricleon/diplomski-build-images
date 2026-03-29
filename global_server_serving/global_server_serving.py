import yaml
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import os
from torchvision import transforms

app = FastAPI()

def get_model_path():
    return f"/home/model/model.pt"

def load_model():
    model_path = get_model_path()
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = cifar10_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
        return JSONResponse({"prediction": int(pred)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    with open("global_server_serving_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    address = config.get("server", {}).get("address", "0.0.0.0:8000")

    if ":" in address:
        host, port_str = address.rsplit(":", 1)
        port = int(port_str)

    uvicorn.run(app, host=host, port=port)
    print(f"Global Server Serving started at {host}:{port}")
