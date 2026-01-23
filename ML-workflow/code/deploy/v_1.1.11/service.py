import bentoml
from bentoml.io import JSON, Text

try:
    from bentoml.models import BentoModel
except Exception as e:
    print(e)
    print("skipping..")
from download_medmnist import preprocess_image_file
import numpy as np
import os
from PIL import Image
import torch

# from torchvision import transforms
import mlflow.pytorch
import requests
from io import BytesIO


# --- Define Service ---
svc = bentoml.Service(name="medmnist_cnn_demo")

# --- Load Model ---
BENTO_MODEL_ARTIFACT = os.environ.get("BENTO_MODEL_ARTIFACT", None)
assert BENTO_MODEL_ARTIFACT is not None, ValueError(
    "BENTO_MODEL_ARTIFACT is not set in environment variables."
)

try:
    bento_model = BentoModel(BENTO_MODEL_ARTIFACT)
except Exception as e:
    print(f"Error loading BentoModel: {e}\n Retrying with bentoml.models.get()")
    try:
        bento_model = bentoml.models.get(BENTO_MODEL_ARTIFACT)
    except Exception as e:
        raise RuntimeError(f"Failed to load BentoModel: {e}")

IMG_SIZE = 28
N_CHANNELS = 3

# --- Load model at startup ---
model_path = bento_model.path_of("mlflow_model")
if torch.cuda.is_available():
    model = mlflow.pytorch.load_model(model_path)
else:
    model = mlflow.pytorch.load_model(model_path, map_location=torch.device("cpu"))

if isinstance(model, torch.nn.Module):
    model.eval()


# --- Define API Endpoint ---
@svc.api(input=Text(), output=JSON())
def predict(input_data: str):
    """
    Takes an image URL, fetches the image, applies preprocessing, and runs inference.
    """
    image_url = input_data

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch image from URL: {e}"}
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    input_tensor = preprocess_image_file(img, img_size=IMG_SIZE, n_channels=N_CHANNELS)

    input_data_np = input_tensor.numpy()

    with torch.no_grad():
        if isinstance(model, torch.nn.Module):
            out = model(input_tensor)
        else:
            out = model.predict(input_data_np)

    if isinstance(out, torch.Tensor):
        probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
    else:
        probs = torch.softmax(torch.from_numpy(out), dim=1).cpu().numpy()[0].tolist()

    pred_idx = int(np.argmax(probs))

    return {
        "input_url": image_url,
        "prediction": pred_idx,
        "probabilities": probs,
    }
