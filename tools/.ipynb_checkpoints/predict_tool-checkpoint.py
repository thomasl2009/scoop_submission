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
        # Load CSV
        df = pd.read_csv(candle_csv_path)

        # Compute features
        features = compute_features(df)
        x = torch.tensor(features).float().unsqueeze(0)  # Add batch dimension

        # Initialize model if not yet done
        if self.model is None:
            input_dim = features.shape[1]
            self.model = CandlePatternPredictor(input_dim)
            self.model.eval()  # Inference mode

        # Run prediction
        with torch.no_grad():
            prob = self.model(x).item()
            
        pred = 1 if prob >= 0.5 else 0
            
        return {
            "probability": prob,
            "prediction": pred,
            "source": candle_csv_path
            }
