import torch
import pandas as pd
from spoon_ai.tools.base import BaseTool
from typing import Any
from model.pytorch_model import CandlePatternPredictor
from model.feature_builder import compute_features

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
        self.model: Any = None  # Will be initialized lazily

    async def execute(self, candle_csv_path: str) -> dict:
        MAX_LEN = 640       # same as before
        FEATURES = 639      # model input size
        
        # Load your 1-row CSV (no label column)
        row = pd.read_csv(f"{candle_csv_path}", header=None).values.flatten()
        
        # Convert to numeric
        row = pd.to_numeric(row, errors='coerce')
        
        # Remove NaNs
        row = row[~np.isnan(row)]
        
        # Convert to float32
        row = row.astype(np.float32)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad to 639 features + 1 label position (but label is missing)
        # Training format = [label | 639 features]
        # Here we only have the 639 features, so if row is shorter → pad to 639
        if len(row) < FEATURES:
            row = np.pad(row, (0, FEATURES - len(row)), constant_values=0)
        
        row = scaler.transform([row])  # keep 2D shape
        tensor_row = torch.tensor(row, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            output = model(tensor_row)           # shape: [1, 2] for 2 classes
            probs = F.softmax(output, dim=1)
        
        percent_class_0 = probs[0][0].item() * 100
        percent_class_1 = probs[0][1].item() * 100
        
        print(f"Class 0 chance: {percent_class_0:.2f}%")
        print(f"Class 1 chance: {percent_class_1:.2f}%")
            
        return {
            "probability": prob,
            "prediction": pred,
            "source": candle_csv_path