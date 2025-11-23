import torch
import pandas as pd
import numpy as np
from spoon_ai.tools.base import BaseTool
from typing import Any
from model.pytorch_model import SimpleModel
import torch.nn.functional as F
import joblib
import os

class CandlePredictionTool(BaseTool):
    name: str = "candle_predictor"
    description: str = "Predicts market reaction from candlestick patterns"

    parameters: dict = {
        "type": "object",
        "properties": {
            "candle_csv_path": {"type": "string", "description": "Path to candlestick CSV file"}
        },
        "required": ["candle_csv_path"]
    }

    def __init__(self):
        super().__init__()
        # Load model
        model_path = os.path.join(os.getcwd(), "path_to_trained_model.pt")
        self.model: Any = SimpleModel()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        # Load scaler
        scaler_path = os.path.join(os.getcwd(), "path_to_scaler.pkl")
        self.scaler = joblib.load(scaler_path)

    async def execute(self, candle_csv_path: str) -> dict:
        print("✅ execute() started")  # Step 0

        FEATURES = 639

        # Load CSV row
        row = pd.read_csv(candle_csv_path, header=None).values.flatten()
        print(f"CSV loaded, shape: {row.shape}")  # Step 1

        row = pd.to_numeric(row, errors='coerce')
        row = row[~np.isnan(row)]
        row = np.nan_to_num(row.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Row after cleaning, length: {len(row)}")  # Step 2

        if len(row) < FEATURES:
            row = np.pad(row, (0, FEATURES - len(row)), constant_values=0)
        print(f"Row after padding, length: {len(row)}")  # Step 3

        # Scale
        row = self.scaler.transform([row])
        print(f"Row after scaling: {row}")  # Step 4

        tensor_row = torch.tensor(row, dtype=torch.float32)
        print(f"Tensor shape: {tensor_row.shape}")  # Step 5

        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_row)
            print(f"Model raw output: {output}")  # Step 6
            probs = F.softmax(output, dim=1)
            print(f"Model probabilities: {probs}")  # Step 7

        pred = int(probs.argmax(dim=1).item())
        prob = float(probs[0][pred].item() * 100)
        print(f"Prediction: {pred}, Probability: {prob}")  # Step 8

        return {
            "prediction": pred,
            "probability": prob,
            "source": candle_csv_path
        }

