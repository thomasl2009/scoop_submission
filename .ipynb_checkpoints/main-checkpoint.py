import asyncio
from agent.agent import CandleAgent
from spoon_ai.chat import ChatBot

async def main():
    agent = CandleAgent(
        llm=ChatBot(
            llm_provider="openai",
            model_name="gpt-5.1"
        )
    )

    # Use your uploaded CSV
    uploaded_path = "/mnt/data/9018a46c-c79d-4f8a-b0ff-ff7888f1ca1d.csv"

    result = await agent.run(
        f"Using CSV file {uploaded_path}, run candle prediction."
    )

    print(result)

if __name__ == "__main__":
    asyncio.run(main())
