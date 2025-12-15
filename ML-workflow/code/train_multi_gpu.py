from accelerate import Accelerator
import argparse
from datetime import datetime
from dotenv import load_dotenv
from download_medmnist import get_datasets
import logging
import mlflow
from mlflow.models import infer_signature
import mlflow.pytorch
from model import SimpleCNN
import numpy as np
import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import refresh_mlflow_token
import pdb


def kickoff_training(args):

    if not args.local_mlflow:
        refresh_mlflow_token()

    # Get the current datetime object
    timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"{timestamp_string}_{args.experiment_name}_bs_{args.batch_size}_lr_{args.lr}"
    )
    experiment_folder = f"experiments/{run_name}"
    os.makedirs(experiment_folder, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device

    # Dataset
    train_ds, val_ds, test_ds, info = get_datasets(
        data_flag=args.data_flag,
        download=True,
        apply_transform=not args.skip_apply_transform,
    )

    logging.info("Dataset ready.")

    num_classes = len(info["label"])
    n_channels = info["n_channels"]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Model, loss, optimizer
    model = SimpleCNN(in_channels=n_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "data_flag": args.data_flag,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "distributed": accelerator.state.num_processes,
            }
        )

        logging.info("Beginning training.")
        ckpt_path = None
        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):

            model.train()
            losses, preds, targets = [], [], []

            for x, y in tqdm(
                train_loader, desc=f"Training Epoch {epoch}/{args.epochs}"
            ):

                if not args.local_mlflow:
                    # Requires refreshing token - auth token expires every 30 min.
                    refresh_mlflow_token(verbose=False)
                x = x.float()
                y = y.squeeze().long()
                out = model(x)
                loss = criterion(out, y)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                losses.append(accelerator.gather(loss.detach()).cpu().numpy())
                preds.extend(accelerator.gather(out.argmax(dim=1)).cpu().numpy())
                targets.extend(accelerator.gather(y).cpu().numpy())
                if args.demo:
                    break
            train_loss = float(np.mean(losses))
            train_acc = float(accuracy_score(targets, preds))

            # Validation
            model.eval()
            v_losses, v_preds, v_targets = [], [], []
            with torch.no_grad():
                for x, y in tqdm(
                    val_loader, desc=f"Validation Epoch {epoch}/{args.epochs}"
                ):
                    x = x.float()
                    y = y.squeeze().long()
                    out = model(x)
                    loss = criterion(out, y)

                    v_losses.append(accelerator.gather(loss.detach()).cpu().numpy())
                    v_preds.extend(accelerator.gather(out.argmax(dim=1)).cpu().numpy())
                    v_targets.extend(accelerator.gather(y).cpu().numpy())
                    if args.demo:
                        break
            val_loss = float(np.mean(v_losses))
            val_acc = float(accuracy_score(v_targets, v_preds))

            # Only log from main process
            if accelerator.is_main_process:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch,
                )

                logging.info(
                    f"Epoch {epoch}/{args.epochs} "
                    f"- train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
                    f"- val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
                )

                if val_acc > best_val_acc:

                    best_val_acc = val_acc
                    ckpt_path = f"{experiment_folder}/best_checkpoint.pth"
                    torch.save(
                        {
                            "model_state": accelerator.get_state_dict(model),
                            "val_acc": val_acc,
                            "epoch": epoch,
                        },
                        ckpt_path,
                    )

                    mlflow.log_artifact(ckpt_path)
        logging.info("Training finished.")
        # ---------- Load Best Model ----------
        if accelerator.is_main_process and ckpt_path and os.path.exists(ckpt_path):
            logging.info("Loading best checkpoint for evaluation...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])

            # ---------- Final Test Eval ----------
            model.to(device)
            model.eval()
            t_preds, t_targets = [], []
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc=f"Test on best model"):
                    x = x.float().to(device)
                    y = y.squeeze().long().to(device)
                    out = model(x)
                    t_preds.extend(out.argmax(dim=1).cpu().numpy())
                    t_targets.extend(y.cpu().numpy())
                    if args.demo:
                        break
            test_acc = float(accuracy_score(t_targets, t_preds))
            logging.info(f"Final Test Accuracy: {test_acc:.4f}")

            mlflow.log_metric("best_val_acc", checkpoint["val_acc"])
            mlflow.log_metric("test_acc", test_acc)

            example_input = torch.randn(
                1, n_channels, args.image_rows, args.image_cols, dtype=torch.float32
            )
            example_output = model(example_input.to(device)).detach().cpu().numpy()
            signature = infer_signature(example_input.cpu().numpy(), example_output)

            try:

                mlflow.pytorch.log_model(
                    model,
                    name="model",
                    signature=signature,
                    registered_model_name=args.registered_model_name or None,
                )
            except Exception as e:
                logging.warning(
                    f"First mlflow.pytorch.log_model failed with exception: {e}\n Trying again..."
                )

                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=args.registered_model_name or None,
                )

            logging.info(f"Model saved in run {run.info.run_id}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip_apply_transform",
        action="store_true",
        help="If set, skips apply transforms to the dataset (default: False)",
    )
    parser.add_argument("--data-flag", type=str, default="pathmnist")
    parser.add_argument("--image-rows", type=int, default=28)
    parser.add_argument("--image-cols", type=int, default=28)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--experiment-name", type=str, default="medmnist_experiment_mlflow"
    )
    parser.add_argument(
        "--local-mlflow",
        action="store_true",
        help="If set, uses local mlflow server",
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default="medmnist_cnn",
        help="Registration model name when logging model in mlflow",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="If set, processes only 1 batch of data",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    # Set MLflow tracking URI
    if args.local_mlflow:
        assert os.path.exists("./.env.local"), FileNotFoundError(
            ".env.local file not found"
        )
        load_dotenv(dotenv_path=".env.local")
    else:
        assert os.path.exists("./.env.pcai"), FileNotFoundError(
            ".env.pcai file not found"
        )
        load_dotenv(dotenv_path=".env.pcai")

    # This is fundamental in order to track training with MLFlow.
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "NA")
    assert MLFLOW_TRACKING_URI != "NA", ValueError(
        "MLFLOW_TRACKING_URI is not set in environment variables."
    )
    # This is fundamental in order to save artifacts into s3 bucket.
    MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "NA")

    logging.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logging.info(f"Using MLFLOW_S3_ENDPOINT_URL: {MLFLOW_S3_ENDPOINT_URL}")

    kickoff_training(args)
