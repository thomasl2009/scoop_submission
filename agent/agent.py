import asyncio
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager
from tools.predict_tool import CandlePredictionTool
from tools.get_candle_data import FetchCandlesTool

class CandleAgent(ToolCallAgent):
    name = "candle_agent"
    description= "Agent that analyzes candle data and predicts market reactions"
    system_prompt = "You are an AI agent that predicts market reactions from candlestick patterns."

    available_tools = ToolManager([
        FetchCandlesTool(),
        CandlePredictionTool()
        # Optional: add tool to fetch candles dynamically via API
    ])

