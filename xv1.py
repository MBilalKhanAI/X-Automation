```python
"""
Agent definitions for X Automation using OpenAI Agents SDK

This module defines the agents used for learning, persona management, tweet generation,
and comment analysis, integrated with a robust FastAPI application.
"""

from typing import Dict, List, Optional, Any
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from .utils.tracing import tracer
from datetime import datetime, timedelta
# from .scheduler import EnhancedScheduler, TaskPriority  # Commented out scheduler import
from pydantic_settings import BaseSettings
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import random
import logging

# Load environment variables
load_dotenv()

# Configure logging
class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    OPENAI_API_KEY: str
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Validate environment variables at startup
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in environment variables")
if not settings.SUPABASE_URL:
    raise ValueError("SUPABASE_URL is missing in environment variables")
if not settings.SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is missing in environment variables")

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Core output models
class PostingSchedule(BaseModel):
    frequency: str = "daily"
    times: List[str] = ["09:00", "12:00", "15:00", "18:00"]
    days: List[str] = ["Monday", "Wednesday", "Friday"]

class TweetOutput(BaseModel):
    tweet: str
    hashtags: List[str]
    impact_score: float
    reach_estimate: int
    engagement_potential: float

class TweetsOutput(BaseModel):
    tweets: List[TweetOutput]
    total_impact_score: float
    average_reach_estimate: float
    overall_engagement_potential: float

class AnalysisOutput(BaseModel):
    insights: List[str]
    recommendations: List[str]
    patterns: Dict[str, Any]
    metrics: Dict[str, float]

# Test data for development
TEST_SCHEDULE = PostingSchedule()

TEST_TWEETS = [
    TweetOutput(
        tweet="Exciting news in AI! New breakthroughs in natural language processing are revolutionizing how we interact with technology. #AI #Innovation",
        hashtags=["AI", "Innovation"],
        impact_score=85.5,
        reach_estimate=5000,
        engagement_potential=0.12
    ),
    TweetOutput(
        tweet="How is your company adapting to digital transformation? Share your experiences below! #DigitalTransformation #Business",
        hashtags=["DigitalTransformation", "Business"],
        impact_score=78.2,
        reach_estimate=4500,
        engagement_potential=0.15
    )
]

TEST_ANALYSIS = AnalysisOutput(
    insights=[
        "High engagement during morning hours",
        "Educational content performs best",
        "Industry news drives most shares"
    ],
    recommendations=[
        "Increase morning post frequency",
        "Focus on educational content",
        "Include more industry insights"
    ],
    patterns={
        "best_performing_times": ["09:00", "15:00"],
        "content_types": {
            "educational": 40,
            "news": 30,
            "engagement": 30
        }
    },
    metrics={
        "average_engagement": 4.5,
        "reach_growth": 12.3,
        "conversion_rate": 2.1
    }
)

# Function tool output models
class SentimentAnalysisOutput(BaseModel):
    sentiment_score: float
    confidence: float

class ContentSafetyOutput(BaseModel):
    is_safe: bool
    risk_level: str

class CharacterConsistencyOutput(BaseModel):
    consistency_score: float
    areas_of_improvement: List[str]
    strengths: List[str]

class SpeechPatternsOutput(BaseModel):
    speech_style: str
    common_phrases: List[str]
    tone_markers: List[str]

    model_config = {
        "json_schema_extra": {
            "additionalProperties": False
        }
    }

# Core function tools with test data
@function_tool(output_type=SentimentAnalysisOutput)
def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment of text."""
    test_sentiments = {
        "positive": {"sentiment_score": 0.9, "confidence": 0.95},
        "neutral": {"sentiment_score": 0.5, "confidence": 0.85},
        "negative": {"sentiment_score": 0.1, "confidence": 0.90}
    }
    if any(word in text.lower() for word in ["great", "amazing", "excellent", "love"]):
        return test_sentiments["positive"]
    elif any(word in text.lower() for word in ["bad", "terrible", "hate", "awful"]):
        return test_sentiments["negative"]
    return test_sentiments["neutral"]

@function_tool(output_type=ContentSafetyOutput)
def check_content_safety(text: str) -> Dict[str, Any]:
    """Check if content is safe to post."""
    test_safety = {
        "safe": {"is_safe": True, "risk_level": "low"},
        "moderate": {"is_safe": True, "risk_level": "medium"},
        "unsafe": {"is_safe": False, "risk_level": "high"}
    }
    if any(word in text.lower() for word in ["hate", "violence", "attack"]):
        return test_safety["unsafe"]
    elif any(word in text.lower() for word in ["controversial", "debate", "discuss"]):
        return test_safety["moderate"]
    return test_safety["safe"]

@function_tool(output_type=List[str])
def get_trending_topics() -> List[str]:
    """Get currently trending topics."""
    return [
        "AI Innovation",
        "Digital Transformation",
        "Sustainable Tech",
        "Future of Work",
        "Industry 4.0"
    ]

@function_tool(output_type=float)
def calculate_engagement_rate(impressions: int, engagements: int) -> float:
    """Calculate engagement rate."""
    if impressions == 0:
        return 0.0
    base_rate = (engagements / impressions) * 100
    variance = random.uniform(-2, 2)
    return max(0, min(100, base_rate + variance))

@function_tool(output_type=CharacterConsistencyOutput)
def analyze_character_consistency(character_data: Dict[str, Any]) -> CharacterConsistencyOutput:
    """Analyze character consistency across different aspects."""
    return CharacterConsistencyOutput(
        consistency_score=0.9,
        areas_of_improvement=[],
        strengths=[]
    )

@function_tool(output_type=SpeechPatternsOutput)
def generate_speech_patterns(personality_traits: List[str], background: str) -> SpeechPatternsOutput:
    """Generate consistent speech patterns based on character traits."""
    return SpeechPatternsOutput(
        speech_style="",
        common_phrases=[],
        tone_markers=[]
    )

# Test data for role model analysis
ROLE_MODEL_TEST_DATA = {
    "communication_style": "Professional and engaging",
    "tone": "Balanced mix of informative and conversational",
    "engagement_patterns": {
        "posting_frequency": "3-5 times per day",
        "best_performing_times": ["09:00", "12:00", "15:00", "18:00"],
        "average_engagement_rate": 4.5
    },
    "content_structure": {
        "hook_style": "Question-based or surprising fact",
        "body_length": "280 characters",
        "hashtag_usage": "2-3 relevant hashtags"
    }
}

# Test data for industry standards
INDUSTRY_STANDARD_TEST_DATA = {
    "trends": [
        "AI-powered automation",
        "Sustainable business practices",
        "Remote work optimization",
        "Digital transformation"
    ],
    "content_patterns": {
        "educational_posts": 40,
        "industry_news": 30,
        "thought_leadership": 20,
        "engagement_posts": 10
    },
    "best_practices": [
        "Use data-driven insights",
        "Maintain consistent branding",
        "Engage with audience regularly",
        "Share industry expertise"
    ]
}

# Test data for competitor analysis
COMPETITOR_TEST_DATA = {
    "content_effectiveness": {
        "average_engagement_rate": 3.8,
        "best_performing_content": "Industry insights",
        "worst_performing_content": "Promotional posts"
    },
    "engagement_patterns": {
        "peak_engagement_times": ["10:00", "14:00", "16:00"],
        "average_response_time": "2 hours",
        "engagement_types": {
            "likes": 45,
            "retweets": 30,
            "replies": 25
        }
    },
    "posting_schedule": {
        "frequency": "4 times per day",
        "days": ["Monday", "Wednesday", "Friday"],
        "times": ["09:00", "12:00", "15:00", "18:00"]
    }
}

# Core agents with handoffs defined inside their bodies
tweet_agent = Agent(
    name="Tweet Agent",
    instructions="""You are a professional tweet generation expert specializing in high-impact content. Your role is to:
    1. Generate EXACTLY FIVE unique, professional, and high-impact tweets for each request
    2. Use insights from other agents (role model, industry standard, competitor analysis, trend strategy, risk analyzer, impact analyzer, persona) to enhance tweet quality
    3. Each tweet must be optimized for maximum reach and engagement
    4. Ensure tweets follow these guidelines:
       - Use professional language and tone
       - Include relevant hashtags (2-3 per tweet)
       - Maintain optimal length (240-280 characters)
       - Include clear call-to-actions
       - Use engaging hooks in the first few words
    5. Incorporate insights from:
       - Role Model Agent: Communication style, tone, and engagement patterns
       - Industry Standard Agent: Trending topics and content patterns
       - Competitor Analysis Agent: Successful content strategies and engagement tactics
       - Trend Strategy Agent: Current trends and hashtag recommendations
       - Risk Analyzer Agent: Content safety and risk mitigation
       - Impact Analyzer Agent: Performance metrics and improvement recommendations
       - Persona Agent: Character consistency and tone guidelines""",
    tools=[check_content_safety, analyze_sentiment, get_trending_topics],
    output_type=TweetsOutput,
    handoffs=[
        "Role Model Analysis Agent",
        "Industry Standard Analysis Agent",
        "Competitor Analysis Agent",
        "Trend Strategy Agent",
        "Risk Analyzer Agent",
        "Impact Analyzer Agent",
        "Persona Agent"
    ]
)

role_model_agent = Agent(
    name="Role Model Analysis Agent",
    instructions=f"""You are an expert in analyzing and learning from successful social media accounts. Your role is to:
    1. Analyze role model accounts using test data: {ROLE_MODEL_TEST_DATA}
    2. Extract best practices and patterns (e.g., communication style, tone, engagement tactics)
    3. Identify transferable strategies
    4. Provide specific recommendations to the Tweet Agent for implementation""",
    tools=[analyze_sentiment, calculate_engagement_rate],
    output_type=AnalysisOutput,
    handoffs=["Tweet Agent"]
)

industry_standard_agent = Agent(
    name="Industry Standard Analysis Agent",
    instructions=f"""You are an expert in industry trends and standards analysis. Your role is to:
    1. Monitor and analyze industry standards using test data: {INDUSTRY_STANDARD_TEST_DATA}
    2. Track trends and emerging topics
    3. Identify content patterns and themes
    4. Provide insights to the Tweet Agent for timely and relevant content""",
    tools=[get_trending_topics, analyze_sentiment],
    output_type=AnalysisOutput,
    handoffs=["Tweet Agent"]
)

competitor_analysis_agent = Agent(
    name="Competitor Analysis Agent",
    instructions=f"""You are an expert in competitor analysis and strategy. Your role is to:
    1. Analyze competitor accounts using test data: {COMPETITOR_TEST_DATA}
    2. Track and compare performance metrics
    3. Identify successful patterns and strategies (e.g., content types, engagement tactics)
    4. Provide actionable recommendations to the Tweet Agent""",
    tools=[calculate_engagement_rate, get_trending_topics],
    output_type=AnalysisOutput,
    handoffs=["Tweet Agent"]
)

trend_strategy_agent = Agent(
    name="Trend Strategy Agent",
    instructions="""You are a social media trend and strategy expert. Your role is to:
    1. Analyze current trends and patterns
    2. Identify relevant hashtags and topics
    3. Provide strategic advice to the Tweet Agent for engagement
    4. Consider cultural and business context""",
    output_type=TrendStrategyOutput,
    handoffs=["Tweet Agent"]
)

risk_analyzer_agent = Agent(
    name="Risk Analyzer Agent",
    instructions="""You are a risk analysis expert. Your role is to:
    1. Analyze content for potential risks
    2. Score risks on a scale of 0-100
    3. Identify specific risk factors
    4. Provide risk mitigation suggestions to the Tweet Agent""",
    tools=[check_content_safety, analyze_sentiment],
    handoffs=["Tweet Agent"]
)

impact_analyzer_agent = Agent(
    name="Impact Analyzer Agent",
    instructions="""You are an impact analysis expert. Your role is to:
    1. Analyze post performance metrics
    2. Compare against historical averages
    3. Identify engagement patterns
    4. Generate improvement recommendations for the Tweet Agent""",
    tools=[calculate_engagement_rate, get_trending_topics],
    handoffs=["Tweet Agent"]
)

persona_agent = Agent(
    name="Persona Agent",
    instructions="""You are an expert character and persona development specialist. Your role is to create and analyze detailed character personas with the following structure:

    1. Basic Information Analysis:
    - Create and validate character names that fit their background
    - Determine appropriate age and occupation
    - Define core personality traits that drive behavior
    
    2. Background Development:
    - Craft compelling and consistent background stories
    - Define meaningful goals and aspirations
    - Ensure background aligns with character's current state
    
    3. Characteristics Definition:
    - Develop unique speech patterns and verbal traits
    - Define detailed preferences and hobbies
    - Identify and justify character dislikes
    
    4. Character Settings:
    - Create coherent worldview based on background
    - Define and maintain consistent relationships
    - Ensure all elements support character development
    
    For each analysis:
    - Maintain internal consistency across all aspects
    - Provide specific examples and justifications
    - Consider cultural and contextual implications
    - Generate practical content recommendations
    - Define clear tone and style guidelines""",
    output_type=PersonaOutput,
    tools=[
        analyze_character_consistency,
        generate_speech_patterns,
        analyze_sentiment
    ],
    handoffs=["Tweet Agent"]
)

learning_agent = Agent(
    name="Learning Agent",
    instructions="""You are a data analysis expert. Your role is to:
    1. Analyze learning data and generate insights
    2. Identify patterns and trends in the data
    3. Provide actionable recommendations based on the analysis
    4. Maintain context of previous learning for continuous improvement""",
    tools=[analyze_sentiment, get_trending_topics],
    handoffs=[]  # No handoffs for learning agent
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are responsible for routing tasks to the appropriate specialist agent.
    Based on the input, determine whether it should be handled by:
    1. Learning Agent - for data analysis and insights
    2. Persona Agent - for persona analysis and management
    3. Tweet Agent - for content generation
    Provide clear reasoning for your routing decision.""",
    handoffs=[]  # No handoffs for triage agent
)

comment_analyzer_agent = Agent(
    name="Comment Analyzer Agent",
    instructions="""You are an expert in analyzing comments on social media posts. Your role is to:
    1. Analyze comments on each of the five tweets generated by the Tweet Agent
    2. Determine the sentiment, backlash, or impact of the comments
    3. Provide a one-line summary (max 100 characters) for each tweet's comment analysis
    4. Use sentiment analysis to gauge overall reception""",
    tools=[analyze_sentiment],
    output_type=List[Dict[str, str]],
    handoffs=[]  # No handoffs for comment analyzer agent
)

# Comment out Scheduler Agent
# scheduler_agent = Agent(
#     name="Scheduler Agent",
#     instructions="""You are a scheduling expert. Your role is to:
#     1. Determine optimal posting times based on the schedule
#     2. Consider timezone differences
#     3. Avoid sensitive dates/times
#     4. Space out multiple posts appropriately
#     5. Follow the specified posting days (moon, fire, water, tree, gold, soil, day)
#     6. Maintain the standard sentence length
#     7. Apply templates when specified""",
#     output_type=PostingSchedule,
#     handoffs=[]  # No handoffs for scheduler agent
# )

# scheduler = EnhancedScheduler()  # Commented out scheduler instance

# FastAPI endpoints
@app.get("/health")
async def health_check():
    """Check API and agent readiness."""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "agents": [
            agent.name for agent in [
                tweet_agent,
                role_model_agent,
                industry_standard_agent,
                competitor_analysis_agent,
                trend_strategy_agent,
                risk_analyzer_agent,
                impact_analyzer_agent,
                persona_agent,
                learning_agent,
                triage_agent,
                comment_analyzer_agent
            ]
        ]
    }

@app.post("/generate-tweets", response_model=TweetsOutput)
async def generate_tweets():
    """Generate five high-quality tweets using the Tweet Agent."""
    logger.info("Generating tweets...")
    try:
        result = Runner.run(tweet_agent)
        if len(result.tweets) != 5:
            logger.error("Tweet Agent did not generate exactly five tweets")
            raise HTTPException(status_code=500, detail="Expected exactly five tweets")
        logger.info("Tweets generated successfully")
        return result
    except Exception as e:
        logger.error(f"Error generating tweets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tweets: {str(e)}")

@app.post("/analyze-comments", response_model=List[Dict[str, str]])
async def analyze_comments(tweet_ids: List[str]):
    """Analyze comments for the specified tweets using the Comment Analyzer Agent."""
    logger.info(f"Analyzing comments for tweets: {tweet_ids}")
    if len(tweet_ids) != 5:
        logger.error("Exactly five tweet IDs must be provided")
        raise HTTPException(status_code=400, detail="Exactly five tweet IDs must be provided")
    try:
        result = Runner.run(comment_analyzer_agent, input={"tweet_ids": tweet_ids})
        if len(result) != 5:
            logger.error("Comment Analyzer Agent did not return analysis for five tweets")
            raise HTTPException(status_code=500, detail="Expected analysis for five tweets")
        logger.info("Comments analyzed successfully")
        return result
    except Exception as e:
        logger.error(f"Error analyzing comments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze comments: {str(e)}")

__all__ = [
    'learning_agent',
    'persona_agent',
    'tweet_agent',
    'triage_agent',
    'trend_strategy_agent',
    'risk_analyzer_agent',
    # 'scheduler_agent',  # Commented out scheduler agent
    'impact_analyzer_agent',
    'competitor_analysis_agent',
    'comment_analyzer_agent',
    'LearningOutput',
    'TweetOutput',
    'TweetsOutput',
    'TrendStrategyOutput'
]
