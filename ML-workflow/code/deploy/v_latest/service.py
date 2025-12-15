import bentoml
from bentoml.models import BentoModel
from download_medmnist import preprocess_image_file
from io import BytesIO
import mlflow.pytorch
import numpy as np
import os
from PIL import Image

# from torchvision import transforms
import requests
import torch


# --- BentoML Service Definition ---
demo_image = bentoml.images.Image(python_version="3.11").python_packages(
    "mlflow", "torch", "Pillow", "torchvision", "numpy", "requests", "medmnist==2.2.0"
)


@bentoml.service(
    image=demo_image,
    resources={"cpu": "1"},
    traffic={"timeout": 60},
)
class MedmnistCNNModel:
    # Model/Artifact Tags
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

    # Image Configuration
    IMG_SIZE = 28
    N_CHANNELS = 3

    def __init__(self):
        # Select runtime device
        self.device = "cpu"
        print(f"[INFO] Using device: {self.device}")

        # Load model via MLflow and move to correct device
        model_path = self.bento_model.path_of("mlflow_model")
        model = mlflow.pytorch.load_model(model_path, map_location=self.device)

        if isinstance(model, torch.nn.Module):
            model.to(self.device)
            model.eval()

        self.model = model

    @bentoml.api
    def predict(self, input_data: str) -> dict:
        """
        Takes an image URL, fetches the image, applies preprocessing, and runs inference.
        """
        image_url = input_data

        try:
            # 1. Fetch the image content using requests
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # 2. Use BytesIO to simulate a file and load the image
            img = Image.open(BytesIO(response.content)).convert("RGB")

        except requests.exceptions.RequestException as e:
            # Handle request errors (e.g., network issue, 404)
            return {"error": f"Failed to fetch image from URL: {e}"}
        except Exception as e:
            # Handle PIL image opening errors
            return {"error": f"Failed to process image: {e}"}

        # Preprocess
        input_tensor = preprocess_image_file(
            img, img_size=self.IMG_SIZE, n_channels=self.N_CHANNELS
        )

        # Convert to numpy for MLflow predict (if needed)
        if self.device == "cpu":
            input_data = input_tensor.numpy()

        # Inference
        with torch.no_grad():
            if hasattr(self.model, "predict"):  # MLflow model wrapper
                out = self.model.predict(input_tensor.cpu().numpy())
                out = torch.from_numpy(np.array(out)).to(self.device)
            else:
                out = self.model(input_tensor)

        # --- Postprocess ---
        probs = torch.softmax(out, dim=1).detach().cpu().numpy()[0].tolist()
        pred_idx = int(np.argmax(probs))

        return {
            "input_url": image_url,
            "prediction": pred_idx,
            "probabilities": probs,
        }
