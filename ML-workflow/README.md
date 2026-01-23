# MLflow + BentoML: End-to-End AI Pipeline Demo

**A complete MLOps pipeline** showcasing how to train, track, and deploy deep learning models using **MLflow**, **BentoML**, and **S3**, integrated with **MLIS** for production deployment.


![workflow](./images/overall_workflow.png)


## Overview

This demo walks through a **production-ready AI workflow**, including:

- Dataset preprocessing (MedMNIST)
- Model training with MLflow tracking
- Multi-GPU acceleration using HuggingFace’s `Accelerate` library
- Best model selection by validation metrics
- Local inference with MLflow
- BentoML packaging for deployment
- Upload to S3 + Deploy on MLIS
- API inference with authentication

---

## Architecture

```
[Dataset] → [MLflow Training + Tracking] → [Best Model] 
→ [BentoML Packaging] → [S3 Upload] → [MLIS Deployment] → [API Inference]
```

---

## Quick Start

### 1️⃣ Install Dependencies
```bash
python -m venv .venv_bento
source .venv_bento/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```bash
python train_multi_gpu.py --epochs 6 --lr 1e-3
```

### 3️⃣ Select the Best Model
```bash
python pick_best.py
```

### 4️⃣ Run Inference
```bash
python infer.py --model-uri models:/medmnist_cnn/3 --file-id 0
```

### 5️⃣ Package and Deploy with BentoML
```bash
cd deploy/v_latest
bentoml build
```

Then upload the `.bento` archive to your S3 bucket and deploy it in **MLIS**.

---

## Online Inference Example

```python
import requests, json

response = requests.post(
    f"{MODEL_URI}/predict",
    headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
    data=json.dumps({"input_data": INPUT_DATA}),
    verify=False
)
print(response.json())
```

---

## Documentation

For the full workflow — including **MLflow setup, BentoML archive fixes, and MLIS deployment instructions** — see:

👉 [**Full Technical Documentation → `README_FULL.md`**](./README_FULL.md)

---

## Credits
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [MLflow](https://mlflow.org/)
- [BentoML](https://bentoml.com/)
- [MedMNIST Dataset](https://medmnist.com/)
- [HPE Private Cloud Solutions](https://www.hpe.com/us/en/private-cloud-solutions.html)

---

## 📜 License
Distributed under the [Apache 2.0](LICENSE) license.
