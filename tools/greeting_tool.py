from spoon_ai.tools.base import BaseTool

class GreetingTool(BaseTool):
    name = "greeting"
    description = "Generate a greeting"

    parameters = {
        "type": "object",
        "properties": { "name": {"type": "string"} },
        "required": ["name"]
    }

    async def execute(self, name: str):
        return f"Hello {name}! Welcome to SpoonOS!"
