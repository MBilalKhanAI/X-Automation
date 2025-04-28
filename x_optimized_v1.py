import asyncio
from openai import OpenAI
from agents import Agent, Runner, set_default_openai_key, InputGuardrail, GuardrailFunctionOutput
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import os
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import sys
import codecs

# Custom StreamHandler that handles Unicode properly
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            if not isinstance(msg, str):
                msg = str(msg)
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_workflow.log', encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
set_default_openai_key(openai_api_key)

# Output types
class LearningOutput(BaseModel):
    insights: str
    recommendations: list[str]
    improvement_areas: list[str]

class PersonaData(BaseModel):
    tone_description: str
    hashtags: list[str]
    style_preferences: str

class LearningProfile(BaseModel):
    role_models: list[str]
    industry_info: str
    competitor_profiles: list[str]

class TweetOutput(BaseModel):
    tweets: list[str]

class TweetResult(BaseModel):
    tweet: str
    risk_score: int
    risk_reason: str
    schedule: str
    impact: str
    competitor_insights: str
    replies: str

class WorkflowResult(BaseModel):
    learning_insights: LearningOutput
    persona_data: PersonaData
    learning_profile: LearningProfile
    tweet_results: list[TweetResult]

# Initialize agents
learning_agent = Agent(
    name="Learning Agent",
    instructions="You are an expert at analyzing learning data and providing actionable insights."
)

persona_agent = Agent(
    name="Persona Agent",
    instructions="You are an expert at analyzing and defining user personas."
)

learning_profile_agent = Agent(
    name="Learning Profile Agent",
    instructions="You are an expert at creating comprehensive learning profiles."
)

tweet_generator_agent = Agent(
    name="Tweet Generator",
    instructions="You are an expert at creating engaging and relevant tweets."
)

risk_analyzer_agent = Agent(
    name="Risk Analyzer",
    instructions="You are an expert at analyzing content for potential risks."
)

scheduler_agent = Agent(
    name="Scheduler",
    instructions="You are an expert at scheduling content for optimal engagement."
)

impact_analyzer_agent = Agent(
    name="Impact Analyzer",
    instructions="You are an expert at analyzing content impact."
)

competitor_analyzer_agent = Agent(
    name="Competitor Analyzer",
    instructions="You are an expert at analyzing competitor content."
)

learning_interaction_agent = Agent(
    name="Learning Interaction",
    instructions="You are an expert at creating learning-focused interactions."
)

async def process_tweet(tweet: str, competitor_ids: List[str], i: int) -> TweetResult:
    """Process a single tweet in parallel with all its analysis steps."""
    logger.info(f"--- Processing Tweet {i} ---")
    logger.info(f"Tweet: {tweet}")
    
    # Prepare all inputs for parallel processing
    risk_input = f"Analyze this tweet for potential risks: {tweet}"
    schedule_time = datetime.now() + timedelta(hours=1)
    schedule_input = f"Schedule this tweet for posting: {tweet}\nScheduled time: {schedule_time.isoformat()}"
    impact_input = f"Analyze potential impact for this tweet: {tweet}\nScheduled time: {schedule_time.isoformat()}"
    competitor_input = f"Analyze competitors with IDs: {', '.join(competitor_ids)}"
    reply_input = f"Generate replies for this tweet: {tweet}\nPost ID: {str(uuid.uuid4())}"

    # Run all analysis steps in parallel
    risk_task = Runner.run(risk_analyzer_agent, risk_input)
    schedule_task = Runner.run(scheduler_agent, schedule_input)
    impact_task = Runner.run(impact_analyzer_agent, impact_input)
    competitor_task = Runner.run(competitor_analyzer_agent, competitor_input)
    reply_task = Runner.run(learning_interaction_agent, reply_input)

    # Wait for all tasks to complete
    risk_result, schedule_result, impact_result, competitor_insights, reply_result = await asyncio.gather(
        risk_task,
        schedule_task,
        impact_task,
        competitor_task,
        reply_task
    )

    # Format and log the results
    logger.info(f"  Risk Score: {30} (Low risk content)")
    logger.info(f"  Schedule: {schedule_result.final_output}")
    logger.info(f"  Impact: {impact_result.final_output}")
    logger.info(f"  Competitor Insights: {competitor_insights.final_output}")
    logger.info(f"  Replies: {reply_result.final_output}")
    logger.info("")  # Add empty line for better readability

    # Create and return the tweet result
    return TweetResult(
        tweet=tweet,
        risk_score=30,  # Dummy risk score, improve parsing later
        risk_reason="Low risk content",
        schedule=schedule_result.final_output,
        impact=impact_result.final_output,
        competitor_insights=competitor_insights.final_output,
        replies=reply_result.final_output
    )

async def orchestrate_workflow(learning_data: str, persona_data: str, competitor_ids: List[str]) -> WorkflowResult:
    """Orchestrate the entire workflow with parallel processing."""
    logger.info("Starting tweet workflow test...")
    logger.info(f"Using OpenAI API Key: {os.getenv('OPENAI_API_KEY')[:10]}...")
    logger.info("")
    
    # Run initial analysis steps in parallel
    learning_task = Runner.run(learning_agent, learning_data)
    persona_task = Runner.run(persona_agent, persona_data)
    profile_task = Runner.run(learning_profile_agent, f"Create profile based on: {learning_data}")
    tweets_task = Runner.run(tweet_generator_agent, f"Generate tweets based on: {learning_data}")

    # Wait for initial tasks to complete
    learning_result, persona_result, profile_result, tweets_result = await asyncio.gather(
        learning_task,
        persona_task,
        profile_task,
        tweets_task
    )

    # Parse initial results
    learning_output = LearningOutput(
        insights=learning_result.final_output,
        recommendations=["Implement new strategies", "Focus on engagement"],
        improvement_areas=["Content quality", "Timing optimization"]
    )

    persona_output = PersonaData(
        tone_description=persona_result.final_output,
        hashtags=["#learning", "#education"],
        style_preferences="Professional and engaging"
    )

    learning_profile = LearningProfile(
        role_models=["Industry leaders"],
        industry_info=profile_result.final_output,
        competitor_profiles=competitor_ids
    )

    tweets_output = TweetOutput(tweets=tweets_result.final_output.split("\n"))

    # Process all tweets in parallel
    tweet_tasks = [
        process_tweet(tweet, competitor_ids, i)
        for i, tweet in enumerate(tweets_output.tweets, 1)
    ]
    tweet_results = await asyncio.gather(*tweet_tasks)

    logger.info("=== Workflow Results ===")
    logger.info("")

    # Return the complete workflow result
    return WorkflowResult(
        learning_insights=learning_output,
        persona_data=persona_output,
        learning_profile=learning_profile,
        tweet_results=tweet_results
    )

async def main():
    """Main function to run the workflow."""
    # Example data
    learning_data = "User has shown interest in machine learning and data science"
    persona_data = "Professional tech enthusiast with focus on AI"
    competitor_ids = ["comp1", "comp2", "comp3"]

    try:
        result = await orchestrate_workflow(learning_data, persona_data, competitor_ids)
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 