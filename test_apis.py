import os
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_environment_variables():
    """Test if all required environment variables are set."""
    required_vars = [
        'OPENAI_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ All required environment variables are set")
    return True

def test_openai_connection():
    """Test OpenAI API connection."""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=5
        )
        logger.info("✅ OpenAI API connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ OpenAI API connection failed: {str(e)}")
        return False

def test_supabase_connection():
    """Test Supabase connection."""
    try:
        supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        # Test a simple query
        response = supabase.table('users').select('count').execute()
        logger.info("✅ Supabase connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {str(e)}")
        return False

def main():
    """Run all API tests."""
    logger.info("Starting API configuration tests...")
    
    # Load environment variables
    load_dotenv()
    
    # Run tests
    env_test = test_environment_variables()
    if not env_test:
        logger.error("❌ Environment variables test failed. Please check your .env file.")
        return
    
    openai_test = test_openai_connection()
    supabase_test = test_supabase_connection()
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Environment Variables: {'✅' if env_test else '❌'}")
    logger.info(f"OpenAI API: {'✅' if openai_test else '❌'}")
    logger.info(f"Supabase: {'✅' if supabase_test else '❌'}")
    
    if all([env_test, openai_test, supabase_test]):
        logger.info("\n🎉 All API tests passed successfully!")
    else:
        logger.error("\n❌ Some API tests failed. Please check the logs above for details.")

if __name__ == "__main__":
    main() 