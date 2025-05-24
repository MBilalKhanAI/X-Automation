import os
from dotenv import load_dotenv
from agents import Agent, Runner, set_default_openai_key
from pydantic import BaseModel
from typing import List, Dict, Any
import shopify
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('product_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Shopify API
shopify.ShopifyResource.set_site(os.getenv('SHOPIFY_STORE_URL'))
shopify.ShopifyResource.set_headers({'X-Shopify-Access-Token': os.getenv('SHOPIFY_ACCESS_TOKEN')})

# Define output types
class ContentAnalysis(BaseModel):
    meta_title: str
    meta_description: str
    content: str
    keywords: List[str]
    product_categories: List[str]

class ProductInfo(BaseModel):
    id: int
    title: str
    description: str
    price: str
    availability: bool
    relevance_score: float

class UpdatedContent(BaseModel):
    original_content: str
    updated_content: str
    linked_products: List[Dict[str, Any]]
    product_placements: List[Dict[str, Any]]

# Create specialized agents
content_analyzer_agent = Agent(
    name="Content Analyzer",
    instructions=(
        "You are a specialized content analysis agent. "
        "Analyze the provided content and extract:\n"
        "1. Meta title\n"
        "2. Meta description\n"
        "3. Key keywords\n"
        "4. Relevant product categories\n\n"
        "Focus on identifying product-related terms and categories that would be relevant for Shopify products."
    ),
    output_type=ContentAnalysis
)

product_matcher_agent = Agent(
    name="Product Matcher",
    instructions=(
        "You are a specialized product matching agent. "
        "Given content analysis and product data, determine:\n"
        "1. Which products are most relevant to the content\n"
        "2. Where in the content these products should be linked\n"
        "3. What brief product information to include\n\n"
        "Only consider products that are currently available in the Shopify store."
    ),
    output_type=UpdatedContent
)

# Retry decorator for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def run_agent_with_retry(agent, input_text):
    try:
        return await Runner.run(agent, input_text)
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise

def fetch_shopify_products():
    """Fetch all available products from Shopify store"""
    try:
        products = shopify.Product.find()
        return [
            {
                "id": product.id,
                "title": product.title,
                "description": product.body_html,
                "price": product.variants[0].price if product.variants else "N/A",
                "availability": product.variants[0].inventory_quantity > 0 if product.variants else False
            }
            for product in products
        ]
    except Exception as e:
        logger.error(f"Error fetching Shopify products: {str(e)}")
        return []

async def process_content(content: str) -> Dict[str, Any]:
    """
    Process content and link relevant Shopify products
    
    Args:
        content (str): The original content to process
        
    Returns:
        Dict[str, Any]: Updated content with linked products
    """
    try:
        # Step 1: Analyze content
        analysis_result = await run_agent_with_retry(
            content_analyzer_agent,
            f"Analyze this content: {content}"
        )
        content_analysis = analysis_result.final_output_as(ContentAnalysis)
        
        # Step 2: Fetch available products
        available_products = fetch_shopify_products()
        
        # Step 3: Match products to content
        matching_input = {
            "content_analysis": content_analysis.model_dump(),
            "available_products": available_products
        }
        
        matching_result = await run_agent_with_retry(
            product_matcher_agent,
            f"Match products to this content analysis: {matching_input}"
        )
        
        return matching_result.final_output_as(UpdatedContent).model_dump()
        
    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        return {
            "error": str(e),
            "original_content": content,
            "updated_content": content,
            "linked_products": [],
            "product_placements": []
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example content
        content = """
        Looking for the perfect summer outfit? Our latest collection features 
        comfortable and stylish clothing perfect for warm weather. From breezy 
        dresses to lightweight tops, we have everything you need to stay cool 
        and fashionable this season.
        """
        
        result = await process_content(content)
        print("\nOriginal Content:")
        print(result["original_content"])
        print("\nUpdated Content:")
        print(result["updated_content"])
        print("\nLinked Products:")
        for product in result["linked_products"]:
            print(f"- {product['title']} (Relevance: {product['relevance_score']})")
    
    asyncio.run(main())
