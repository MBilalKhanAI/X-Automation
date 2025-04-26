import asyncio
from openai import OpenAI
from agents import Agent, Runner, set_default_openai_key, InputGuardrail, GuardrailFunctionOutput
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
from pydantic import BaseModel
from dotenv import load_dotenv

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
    competitor_profiles: list[dict]

class CharacterProfile(BaseModel):
    name: str
    age: int
    occupation: str
    personality: List[str]
    upbringing: str
    goals: str
    speech_traits: str
    preferences: List[str]
    dislikes: List[str]
    worldview: str
    relationships: str

class TweetOutput(BaseModel):
    tweet: str

class TrendStrategyOutput(BaseModel):
    trending_topics: List[str]
    recommended_hashtags: List[str]
    strategy_advice: str

class TweetsOutput(BaseModel):
    tweets: List[str]

# Agents
learning_interaction_agent = Agent(
    name="Learning Interaction Analyzer",
    instructions="You are a specialized learning and interaction analysis agent. Analyze the learning profile and provide insights, recommendations, and improvement areas. Consider Japanese business context and cultural nuances.",
    output_type=LearningOutput
)

content_generator_agent = Agent(
    name="Persona-based Tweet Generator",
    instructions=(
        "You are a professional social media content creator for Japanese enterprises. "
        "You always generate impressive, engaging, and professional tweets. "
        "You must use the following character profile as context for every tweet you generate. "
        "The tweet should reflect the character's unique background, personality, goals, and worldview. "
        "Make sure the tweet is not generic or unimpressive, but instead feels authentic to the character. "
        "If possible, incorporate their speech traits, preferences, and relationships. "
        "Keep the tweet formal, professional, and suitable for a Japanese business audience. "
        "Do not mention the character's profile directly; instead, let it inform the style and content of the tweet."
        "Generate exactly 5 impressive, engaging, and professional tweets as a list. "
        "Each tweet should be unique, not generic, and reflect the character's profile. "
        "Return the tweets as a JSON list under the key 'tweets'."
    ),
    output_type=TweetsOutput
)

risk_analyzer_agent = Agent(
    name="Tweet Risk Analyzer",
    instructions=(
        "You are a specialized risk analysis agent for social media content. "
        "Analyze tweets for potential risks and backlash, score risks on a scale of 0-100, "
        "consider cultural, social, and business impacts, identify specific risk factors and reasons, "
        "provide risk mitigation suggestions, and flag high-risk content for review."
    ),
    handoff_description="Specialist agent for analyzing tweet risks and potential backlash"
)

scheduler_agent = Agent(
    name="Post Timing Optimizer",
    instructions=(
        "You are an intelligent post scheduling optimizer. "
        "Determine optimal posting times based on audience analytics, consider timezone differences (especially for Japan), "
        "avoid scheduling during sensitive dates/times, space out multiple posts appropriately, "
        "schedule follow-up analysis checks (24h post-publishing), and consider peak engagement times for Japanese business audience."
    ),
    handoff_description="Specialist agent for optimizing post timing and scheduling"
)

impact_analyzer_agent = Agent(
    name="Engagement Impact Analyzer",
    instructions=(
        "You are a post-performance and impact analysis specialist. "
        "Analyze 24-hour post performance metrics, compare metrics against historical averages, "
        "identify engagement patterns, analyze sentiment of responses, track hashtag performance, "
        "generate improvement recommendations, and monitor for any negative trends or backlash."
    ),
    handoff_description="Specialist agent for analyzing post impact and performance"
)

competitor_analyzer_agent = Agent(
    name="Competitor Intelligence Analyzer",
    instructions=(
        "You are a competitor analysis specialist focusing on top-performing accounts. "
        "Analyze successful competitor posts, identify trending topics in the industry, extract effective hashtag strategies, "
        "analyze posting patterns and timing, identify engagement tactics, compare performance metrics, and generate strategic recommendations."
    ),
    handoff_description="Specialist agent for analyzing competitor strategies and performance"
)

trend_strategy_agent = Agent(
    name="Trend Strategy Agent",
    instructions=(
        "You are a social media trend and strategy expert for Japanese enterprises. "
        "Given a character profile and recent social media data, you analyze current trends, "
        "suggest relevant hashtags, and provide strategic advice for maximizing engagement. "
        "Your recommendations should be professional, culturally appropriate, and tailored to the character's persona and business goals."
    ),
    output_type=TrendStrategyOutput
)

# Set up handoffs between agents
content_generator_agent.handoffs = [risk_analyzer_agent]
risk_analyzer_agent.handoffs = [scheduler_agent]
scheduler_agent.handoffs = [impact_analyzer_agent]
impact_analyzer_agent.handoffs = [competitor_analyzer_agent]

# Orchestration logic
async def orchestrate_workflow(
    user_id: int,
    persona_data: PersonaData,
    account_id: str,
    competitor_ids: list[str],
    learning_profile: LearningProfile
):
    try:
        # Step 1: Validate the learning profile
        learning_result = await Runner.run(
            learning_interaction_agent,
            "Analyze this learning profile and provide recommendations: " + learning_profile.model_dump_json()
        )
        learning_output = learning_result.final_output_as(LearningOutput)

        # Step 2: Generate strategies
        trend_input = (
            f"Persona Data: {persona_data.model_dump_json()}\n"
            f"Learning Output: {learning_output.model_dump_json()}\n"
            "Analyze the persona and learning output, and provide trending topics, recommended hashtags, and strategy advice."
        )
        trend_data = await Runner.run(trend_strategy_agent, trend_input)

        # Step 3: Generate tweets
        tweet_data = {
            "persona": persona_data.tone_description,
            "character_details": persona_data.style_preferences,
            "hashtags": persona_data.hashtags,
            "account_id": account_id
        }
        tweet_input = f"Generate tweets based on the following data: {tweet_data}"
        tweet_result = await Runner.run(content_generator_agent, tweet_input)
        tweets_output = tweet_result.final_output_as(TweetsOutput)

        all_tweet_results = []
        for i, tweet in enumerate(tweets_output.tweets, 1):
            print(f"\n--- Processing Tweet {i} ---")
            print(f"Tweet: {tweet}")

            # Step 4: Risk assessment
            risk_input = f"Analyze this tweet for potential risks: {tweet}"
            risk_result = await Runner.run(risk_analyzer_agent, risk_input)
            # Here, you should parse the risk_result to get the actual risk score and reason
            # For now, let's use a dummy value:
            risk_score = 30
            risk_reason = "Low risk content"

            # Step 5: Schedule
            schedule_time = datetime.now() + timedelta(hours=1)
            schedule_input = f"Schedule this tweet for posting: {tweet}\nScheduled time: {schedule_time.isoformat()}"
            schedule_result = await Runner.run(scheduler_agent, schedule_input)

            # Step 6: Impact analysis
            impact_input = f"Analyze potential impact for this tweet: {tweet}\nScheduled time: {schedule_time.isoformat()}"
            impact_result = await Runner.run(impact_analyzer_agent, impact_input)

            # Step 7: Competitor analysis
            competitor_input = f"Analyze competitors with IDs: {', '.join(competitor_ids)}"
            competitor_insights = await Runner.run(competitor_analyzer_agent, competitor_input)

            # Step 8: Generate replies
            reply_input = f"Generate replies for this tweet: {tweet}\nPost ID: {str(uuid.uuid4())}"
            reply_result = await Runner.run(learning_interaction_agent, reply_input)

            all_tweet_results.append({
                "tweet": tweet,
                "risk_score": risk_score,
                "risk_reason": risk_reason,
                "schedule": schedule_result.final_output,
                "impact": impact_result.final_output,
                "competitor_insights": competitor_insights.final_output,
                "replies": reply_result.final_output
            })

        return {
            "learning_analysis": learning_output,
            "trend_strategy": trend_data.final_output,
            "tweets": all_tweet_results
        }
    except Exception as e:
        print(f"Error in workflow orchestration: {str(e)}")
        raise

# Simple test function
async def test_agent():
    try:
        # Create test data
        learning_profile = LearningProfile(
            role_models=["@BusinessLeader1", "@IndustryExpert2"],
            industry_info="Technology and Innovation",
            competitor_profiles=[
                {"id": "comp1", "name": "Competitor 1"},
                {"id": "comp2", "name": "Competitor 2"}
            ]
        )

        # Run the agent with proper input format
        result = await Runner.run(
            learning_interaction_agent,
            "Analyze this learning profile and provide recommendations: " + learning_profile.model_dump_json()
        )

        # Print the result
        print("\nAgent Response:")
        print(result.final_output)
        
        # Try to parse as LearningOutput
        parsed_output = result.final_output_as(LearningOutput)
        print("\nParsed Output:")
        print("Insights:", parsed_output.insights)
        print("Recommendations:", parsed_output.recommendations)
        print("Improvement Areas:", parsed_output.improvement_areas)

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

# Run workflow
if __name__ == "__main__":
    print("Starting tweet workflow test...")
    print(f"Using OpenAI API Key: {os.getenv('OPENAI_API_KEY')[:8]}...")

    # Example test data
    persona_data = PersonaData(
        tone_description="Professional, innovative, and concise.",
        hashtags=["#Innovation", "#Tech", "#Leadership"],
        style_preferences="Formal, motivational, and forward-thinking."
    )
    learning_profile = LearningProfile(
        role_models=["@BusinessLeader1", "@IndustryExpert2"],
        industry_info="Technology and Innovation",
        competitor_profiles=[
            {"id": "comp1", "name": "Competitor 1"},
            {"id": "comp2", "name": "Competitor 2"}
        ]
    )
    account_id = "main_account"
    competitor_ids = ["comp1", "comp2", "comp3"]

    async def run_workflow():
        result = await orchestrate_workflow(
            user_id=1,
            persona_data=persona_data,
            account_id=account_id,
            competitor_ids=competitor_ids,
            learning_profile=learning_profile
        )
        print("\n=== Workflow Results ===")
        for i, tweet_info in enumerate(result["tweets"], 1):
            print(f"\nTweet {i}: {tweet_info['tweet']}")
            print(f"  Risk Score: {tweet_info['risk_score']} ({tweet_info['risk_reason']})")
            print(f"  Schedule: {tweet_info['schedule']}")
            print(f"  Impact: {tweet_info['impact']}")
            print(f"  Competitor Insights: {tweet_info['competitor_insights']}")
            print(f"  Replies: {tweet_info['replies']}")

    asyncio.run(run_workflow())