from typing import Any
import random 
from mcp.server.fastmcp import FastMCP

#init fastmcp server
mcp = FastMCP("crazy-weather")

@mcp.tool()
def get_weather(location: str) -> dict[str, Any]:
	"""Get current weather for a location"""

    conditions = ["sunny", "cloudy", "rainy", "windy", "stormy"]
    temperature = random.randint(-10, 40)
    return {
        "location": location,
        "temperature": f"{temperature}°C",
        "condition": random.choice(conditions),
    }

mcp.run()