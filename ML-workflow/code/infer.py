import io
import os
import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import numpy as np
import torch
from accelerate import Accelerator
import mlflow.pytorch
from medmnist.dataset import INFO
from download_medmnist import preprocess_image_file, get_datasets
from utils import refresh_mlflow_token
from dotenv import load_dotenv

# -----------------------------
# CONFIG / MLflow
# -----------------------------
LOCAL_MLFLOW = os.environ.get("LOCAL_MLFLOW", True)
MODEL_URI = os.environ.get("MODEL_URI", "models:/medmnist_cnn/2")

DATA_FLAG = os.environ.get("DATA_FLAG", "pathmnist")

# Set MLflow tracking URI
if (isinstance(LOCAL_MLFLOW, str) and LOCAL_MLFLOW == "True") or (
    isinstance(LOCAL_MLFLOW, bool) and LOCAL_MLFLOW
):
    assert os.path.exists("./.env.local"), FileNotFoundError(
        ".env.local file not found"
    )
    load_dotenv(dotenv_path=".env.local")
else:
    assert os.path.exists("./.env.pcai"), FileNotFoundError(".env.pcai file not found")
    load_dotenv(dotenv_path=".env.pcai")
    refresh_mlflow_token()

# This is fundamental in order to track training with MLFlow.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "NA")
# This is fundamental in order to save artifacts into s3 bucket.
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "NA")

assert MLFLOW_TRACKING_URI != "NA", ValueError(
    "MLFLOW_TRACKING_URI is not set in environment variables."
)

print("Using MODEL_URI:", MODEL_URI)
print("Using MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("Using MLFLOW_S3_ENDPOINT_URL:", MLFLOW_S3_ENDPOINT_URL)

# Dataset info
info = INFO[DATA_FLAG]
IMG_SIZE = 28
N_CHANNELS = info["n_channels"]
NUM_CLASSES = len(info["label"])

# Accelerator / device
accelerator = Accelerator()
device = accelerator.device

# FastAPI app
app = FastAPI(title="MedMNIST Model Inference")
model = None


# -----------------------------
# MODEL LOADING
# -----------------------------
def load_model(model_uri: str, device):
    print("Loading model from:", model_uri)
    try:
        m = mlflow.pytorch.load_model(model_uri, map_location=device)
        m = accelerator.prepare(m)
        m.eval()
        print("Model loaded successfully")
        return m
    except Exception as e:
        print("Failed to load model:", e)
        return None


@app.on_event("startup")
async def startup_event():
    """Load model when FastAPI starts"""
    global model
    model_uri = os.environ.get("MODEL_URI", MODEL_URI)
    model = load_model(model_uri, device)


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def main_page():
    return """
    <html>
      <head><title>MEDMNIST Model Inference</title></head>
      <body>
        <h1>Upload image to run inference</h1>
        <form action="/predict_from_filepath" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept="image/*">
          <input type="submit" value="Predict">
        </form>
      </body>
    </html>
    """


@app.post("/predict_from_filepath")
async def predict_from_filepath(file: UploadFile = File(...)):
    global model

    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB" if N_CHANNELS == 3 else "L")

    # Preprocess
    if np.array(img).max() > 128:
        x = preprocess_image_file(img, img_size=IMG_SIZE, n_channels=N_CHANNELS).to(
            device
        )
    else:
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
        pred_idx = int(np.argmax(probs))

    return {"predicted_label": pred_idx, "probabilities": probs}


@app.post("/predict_from_file_id")
async def predict_from_file_id(file_id: int = Form(...)):

    global model
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    _, _, test_ds, _ = get_datasets(DATA_FLAG, apply_transform=True, download=False)

    if file_id < 0 or file_id >= len(test_ds):
        return JSONResponse(
            {"error": f"file_id {file_id} out of range"}, status_code=400
        )

    img, y = test_ds[file_id]

    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    with torch.no_grad():

        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
        pred_idx = int(np.argmax(probs))

    return {"predicted_label": pred_idx, "probabilities": probs, "target_label": int(y)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", type=str, default=MODEL_URI)
    parser.add_argument("--file-path", type=str, default=None)
    parser.add_argument("--file-id", type=int, default=None)
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_uri, device)

    # CLI prediction
    async def run_cli():
        if args.file_path:

            class DummyUploadFile:
                def __init__(self, path):
                    self.path = path

                async def read(self):
                    with open(self.path, "rb") as f:
                        return f.read()

            response = await predict_from_filepath(DummyUploadFile(args.file_path))
            print(response)
        elif args.file_id is not None:
            response = await predict_from_file_id(args.file_id)
            print(response)
        else:
            print("No file_path or file_id provided")

    asyncio.run(run_cli())
