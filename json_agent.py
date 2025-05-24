from agents import Agent, Runner
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define output type using Pydantic
class StructuredResponse(BaseModel):
    summary: str
    key_points: List[str]
    metadata: Dict[str, Any]
    confidence_score: float

    model_config = {
        "json_schema_extra": {
            "type": "object",
            "required": ["summary", "key_points", "metadata", "confidence_score"]
        }
    }

# Create the agent
json_agent = Agent(
    name="JSON Response Agent",
    instructions=(
        "You are a specialized agent that provides structured responses in JSON format. "
        "For any input, you must:\n"
        "1. Generate a concise summary\n"
        "2. Extract 3-5 key points\n"
        "3. Include relevant metadata\n"
        "4. Provide a confidence score (0-1)\n\n"
        "Your response must be in valid JSON format and include all required fields. "
        "The response should be professional, accurate, and well-structured."
    ),
    output_type=StructuredResponse
)

# Example usage function
async def get_structured_response(query: str) -> Dict[str, Any]:
    """
    Get a structured JSON response from the agent.
    
    Args:
        query (str): The input query to process
        
    Returns:
        Dict[str, Any]: The structured response
    """
    try:
        result = await Runner.run(json_agent, query)
        return result.final_output.model_dump()
    except Exception as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example query
        query = "Analyze the impact of artificial intelligence on modern business practices"
        
        # Get response
        response = await get_structured_response(query)
        print("Structured Response:")
        print(response)
    
    # Run the example
    asyncio.run(main()) 