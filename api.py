from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from product_fetch import process_content
import logging
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Product Fetch API",
    description="API for analyzing content and linking Shopify products",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request/Response models
class ContentRequest(BaseModel):
    content: str
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None

class ProductPlacement(BaseModel):
    product_id: int
    product_title: str
    product_description: str
    placement_position: int
    relevance_score: float

class ContentResponse(BaseModel):
    original_content: str
    updated_content: str
    linked_products: List[Dict[str, Any]]
    product_placements: List[ProductPlacement]
    error: Optional[str] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Main content processing endpoint
@app.post("/process-content", response_model=ContentResponse)
async def process_content_endpoint(request: ContentRequest):
    try:
        logger.info("Received content processing request")
        
        # Process the content
        result = await process_content(request.content)
        
        # Convert product placements to the correct format
        product_placements = [
            ProductPlacement(
                product_id=placement["product_id"],
                product_title=placement["product_title"],
                product_description=placement["product_description"],
                placement_position=placement["placement_position"],
                relevance_score=placement["relevance_score"]
            )
            for placement in result.get("product_placements", [])
        ]
        
        return ContentResponse(
            original_content=result["original_content"],
            updated_content=result["updated_content"],
            linked_products=result["linked_products"],
            product_placements=product_placements,
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing content: {str(e)}"
        )

# Example usage endpoint
@app.get("/example")
async def get_example():
    example_content = """
    Looking for the perfect summer outfit? Our latest collection features 
    comfortable and stylish clothing perfect for warm weather. From breezy 
    dresses to lightweight tops, we have everything you need to stay cool 
    and fashionable this season.
    """
    
    try:
        result = await process_content(example_content)
        return {
            "example_content": example_content,
            "processed_result": result
        }
    except Exception as e:
        logger.error(f"Error in example endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating example: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 