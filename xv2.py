import asyncio
import os
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from agents import Agent, Runner, set_default_openai_key, GuardrailFunctionOutput
import sys

# Configuration class for environment variables
class Config:
    def __init__(self):
        try:
            # Try to load .env file with explicit encoding
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except FileNotFoundError:
            print("Warning: .env file not found. Using environment variables if available.")
        except Exception as e:
            print(f"Warning: Error loading .env file: {str(e)}")
            print("Using environment variables if available.")

        # Get API keys with fallbacks
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        # Validate required API keys
        if not self.openai_api_key:
            print("Error: OPENAI_API_KEY is required but not found.")
            print("Please set it in your .env file or environment variables.")
            sys.exit(1)
        
        # Initialize OpenAI
        try:
            set_default_openai_key(self.openai_api_key)
        except Exception as e:
            print(f"Error initializing OpenAI: {str(e)}")
            sys.exit(1)

# Initialize configuration
try:
    config = Config()
except Exception as e:
    print(f"Error initializing configuration: {str(e)}")
    sys.exit(1)

# Optimized logging setup with async handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File handler with buffering
file_handler = logging.FileHandler('twitter_automation.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Output types
class LearningOutput(BaseModel):
    insights: str
    recommendations: list[str]
    improvement_areas: list[str]

class PersonaData(BaseModel):
    tone_description: str
    hashtags: list[str]
    style_preferences: str

class AccountReference(BaseModel):
    account_id: str
    name: str
    description: str
    category: str
    website: str = ""

class LearningProfile(BaseModel):
    role_models: list[AccountReference]
    industry_standards: list[AccountReference]
    competitors: list[AccountReference]
    industry_info: str
    content_guidelines: str
    risk_threshold: int = 50

class TweetOutput(BaseModel):
    tweet: str

class SchedulingPreferences(BaseModel):
    posting_frequency: str
    pre_create_posts: bool
    interactive_format: bool
    auto_post_mode: bool
    posting_days: list[str]
    posting_times: list[int]
    average_sentence_length: str
    reply_templates: 'ReplyTemplate'

class ReplyTemplate(BaseModel):
    comment_template: str
    post_template: str
    reply_template: str
    target_hashtags: list[str]

class EnhancedLearningOutput(BaseModel):
    insights: str
    recommendations: list[str]
    improvement_areas: list[str]
    role_model_analysis: Dict[str, str]
    industry_standards_analysis: Dict[str, str]
    competitor_analysis: Dict[str, str]

    model_config = {
        "extra": "forbid",  # Prevent additional properties
        "json_schema_extra": {
            "type": "object",
            "required": [
                "insights",
                "recommendations",
                "improvement_areas",
                "role_model_analysis",
                "industry_standards_analysis",
                "competitor_analysis"
            ]
        }
    }

class EnhancedTweetOutput(BaseModel):
    tweets: List[str]
    quality_metrics: Dict[str, Any]
    engagement_predictions: Dict[str, Any]
    risk_assessment: Dict[str, Any]

    model_config = {
        "extra": "forbid",  # Prevent additional properties
        "json_schema_extra": {
            "type": "object",
            "required": [
                "tweets",
                "quality_metrics",
                "engagement_predictions",
                "risk_assessment"
            ]
        }
    }

class ContentGuardrailOutput(BaseModel):
    passed: bool
    issues: list[str]
    suggestions: list[str]
    risk_score: int

class EngagementMetrics(BaseModel):
    likes: int
    retweets: int
    replies: int
    impressions: int
    engagement_rate: float

class ContentAnalysisOutput(BaseModel):
    sentiment: str
    keywords: list[str]
    hashtag_performance: Dict[str, float]
    audience_reaction: str
    improvement_suggestions: list[str]

class TrendStrategyOutput(BaseModel):
    trending_topics: List[str]
    recommended_hashtags: List[str]
    strategy_advice: str

class TweetsOutput(BaseModel):
    tweets: List[str]

class SafetySettings(BaseModel):
    role_models: list[AccountReference]
    industry_standards: list[AccountReference]
    competitors: list[AccountReference]
    content_guidelines: str
    risk_threshold: int = 50

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

# Optimized guardrail functions
class GuardrailPatterns:
    def __init__(self):
        self.sensitive_terms = re.compile(r'\b(war|politics|religion)\b', re.IGNORECASE)
        self.url_pattern = re.compile(r'http[s]?://')

guardrail_patterns = GuardrailPatterns()

def content_quality_guardrail(content: str) -> GuardrailFunctionOutput:
    issues = []
    suggestions = []
    
    if len(content) > 280:
        issues.append("Tweet exceeds 280 characters")
        suggestions.append("Shorten the tweet to fit within Twitter's limit")
    
    hashtag_count = content.count('#')
    if hashtag_count > 3:
        issues.append(f"Too many hashtags ({hashtag_count})")
        suggestions.append("Limit hashtags to 2-3 per tweet")
    
    if guardrail_patterns.url_pattern.search(content):
        issues.append("Contains URL")
        suggestions.append("Ensure URL is relevant and trustworthy")
    
    return GuardrailFunctionOutput(
        passed=len(issues) == 0,
        issues=issues,
        suggestions=suggestions
    )

def cultural_sensitivity_guardrail(content: str) -> GuardrailFunctionOutput:
    issues = []
    suggestions = []
    
    matches = guardrail_patterns.sensitive_terms.findall(content)
    for term in matches:
        issues.append(f"Contains potentially sensitive term: {term}")
        suggestions.append(f"Review context of '{term}' usage")
    
    return GuardrailFunctionOutput(
        passed=len(issues) == 0,
        issues=issues,
        suggestions=suggestions
    )

# Agents
learning_interaction_agent = Agent(
    name="Learning Interaction Analyzer",
    instructions=(
        "You are a specialized learning and interaction analysis agent. "
        "Analyze the learning profile and provide comprehensive insights based on:\n"
        "1. Role Models: Analyze communication style, tone, and engagement strategies\n"
        "2. Industry Standards: Evaluate content quality and trend alignment\n"
        "3. Competitors: Assess content strategy and audience engagement\n\n"
        "Provide detailed analysis for each category and specific recommendations. "
        "Consider Japanese business context and cultural nuances."
    ),
    output_type=EnhancedLearningOutput
)

content_generator_agent = Agent(
    name="Persona-based Tweet Generator",
    instructions=(
        "You are a professional social media content creator for Japanese enterprises. "
        "Generate tweets considering:\n"
        "1. Role Models: Emulate successful communication styles\n"
        "2. Industry Standards: Incorporate relevant trends\n"
        "3. Competitors: Differentiate while learning from successes\n\n"
        "For each tweet, provide:\n"
        "- Quality metrics (readability, engagement potential)\n"
        "- Engagement predictions\n"
        "- Risk assessment\n\n"
        "Generate exactly 5 impressive, engaging, and professional tweets. "
        "Each tweet should be unique and reflect the character's profile. "
        "Return the tweets with their metrics as a structured JSON."
    ),
    output_type=EnhancedTweetOutput
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
        "You are an intelligent post scheduling optimizer for Japanese social media. "
        "Based on the provided scheduling preferences, determine the optimal posting schedule. "
        "Consider the following parameters:\n"
        "- Posting frequency (e.g., 1 day)\n"
        "- Pre-creation of posts\n"
        "- Interactive format requirements\n"
        "- Auto post mode settings\n"
        "- Specific posting days (moon, fire, water, tree, gold, soil, day)\n"
        "- Posting times (0-23 hours)\n"
        "- Average sentence length (標準 or other values)\n"
        "- Reply templates and target hashtags\n\n"
        "Generate a detailed schedule that:\n"
        "1. Respects the posting frequency\n"
        "2. Aligns with the specified days and times\n"
        "3. Considers Japanese time zones and business hours\n"
        "4. Optimizes for maximum engagement\n"
        "5. Includes follow-up analysis timing\n"
        "6. Incorporates reply templates and hashtags\n"
        "Return the schedule in a structured format with specific dates and times."
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

# Handoffs
content_generator_agent.handoffs = [risk_analyzer_agent]
risk_analyzer_agent.handoffs = [scheduler_agent]
scheduler_agent.handoffs = [impact_analyzer_agent]
impact_analyzer_agent.handoffs = [competitor_analyzer_agent]

# Optimized orchestration
async def orchestrate_workflow(
    user_id: int,
    persona_data: PersonaData,
    account_id: str,
    learning_profile: LearningProfile,
    scheduling_preferences: SchedulingPreferences
) -> Dict[str, Any]:
    workflow_results = {
        "success": False,
        "error": None,
        "tweets": [],
        "analysis": {}
    }
    
    try:
        logger.info(f"Starting workflow for user {user_id}", extra={"user_id": user_id})
        
        # Pre-serialize learning profile
        learning_input = learning_profile.model_dump()
        learning_input_str = (
            f"Analyze this learning profile and provide comprehensive recommendations:\n"
            f"Role Models: {[rm['account_id'] for rm in learning_input['role_models']]}\n"
            f"Industry Standards: {[is_['account_id'] for is_ in learning_input['industry_standards']]}\n"
            f"Competitors: {[comp['account_id'] for comp in learning_input['competitors']]}\n"
            f"Industry Info: {learning_input['industry_info']}\n"
            f"Content Guidelines: {learning_input['content_guidelines']}\n"
            f"Risk Threshold: {learning_input['risk_threshold']}"
        )
        
        # Run learning analysis
        learning_result = await Runner.run(learning_interaction_agent, learning_input_str)
        learning_output = learning_result.final_output_as(EnhancedLearningOutput)
        logger.info("Learning analysis completed", extra={"user_id": user_id})
        
        # Prepare tweet input
        tweet_data = {
            "persona": persona_data.tone_description,
            "character_details": persona_data.style_preferences,
            "hashtags": persona_data.hashtags,
            "account_id": account_id,
            "learning_insights": learning_output.model_dump()
        }
        tweet_input = f"Generate tweets based on the following data: {tweet_data}"
        
        # Run content generation
        tweet_result = await Runner.run(content_generator_agent, tweet_input)
        tweets_output = tweet_result.final_output_as(EnhancedTweetOutput)
        
        # Store results
        for i, tweet in enumerate(tweets_output.tweets):
            workflow_results["tweets"].append({
                "content": tweet,
                "quality_metrics": tweets_output.quality_metrics.get(f"tweet_{i+1}", {}),
                "engagement_predictions": tweets_output.engagement_predictions.get(f"tweet_{i+1}", {}),
                "risk_assessment": tweets_output.risk_assessment.get(f"tweet_{i+1}", {})
            })
        
        logger.info(f"Generated {len(tweets_output.tweets)} tweets", extra={"user_id": user_id})
        workflow_results["success"] = True
        workflow_results["analysis"] = learning_output.model_dump()
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}", extra={"user_id": user_id})
        workflow_results["error"] = str(e)
    
    return workflow_results

# Optimized test function
async def test_workflow():
    try:
        logger.info("Starting workflow test")
        
        # Shared test data
        role_models = [
            AccountReference(
                account_id="@BusinessLeader1",
                name="Business Leader 1",
                description="Innovative business leader in technology",
                category="role_model"
            ),
            AccountReference(
                account_id="@IndustryExpert2",
                name="Industry Expert 2",
                description="Expert in Japanese business culture",
                category="role_model"
            )
        ]
        industry_standards = [
            AccountReference(
                account_id="@TechTrends1",
                name="Tech Trends",
                description="Latest technology trends",
                category="industry_standard",
                website="https://www.techtrends.com"
            )
        ]
        competitors = [
            AccountReference(
                account_id="@Competitor1",
                name="Competitor 1",
                description="Direct competitor in technology sector",
                category="competitor"
            )
        ]
        
        learning_profile = LearningProfile(
            role_models=role_models,
            industry_standards=industry_standards,
            competitors=competitors,
            industry_info="Technology and Innovation",
            content_guidelines="Professional, innovative, and culturally appropriate",
            risk_threshold=50
        )
        
        persona_data = PersonaData(
            tone_description="Professional, innovative, and concise.",
            hashtags=["#Innovation", "#Tech", "#Leadership"],
            style_preferences="Formal, motivational, and forward-thinking."
        )
        
        reply_templates = ReplyTemplate(
            comment_template="Thank you for your comment! We appreciate your thoughts on {topic}.",
            post_template="New post: {content} #Innovation #Tech",
            reply_template="We're glad you found this interesting! Here's more information: {additional_info}",
            target_hashtags=["#Innovation", "#Tech", "#Leadership"]
        )
        
        scheduling_preferences = SchedulingPreferences(
            posting_frequency="1 day",
            pre_create_posts=True,
            interactive_format=True,
            auto_post_mode=True,
            posting_days=["moon", "fire", "water", "tree", "gold", "soil", "day"],
            posting_times=list(range(24)),
            average_sentence_length="標準",
            reply_templates=reply_templates
        )
        
        account_id = "main_account"
        
        # Run learning analysis and workflow
        learning_task = Runner.run(
            learning_interaction_agent,
            "Analyze this learning profile and provide recommendations: " + learning_profile.model_dump_json()
        )
        workflow_task = orchestrate_workflow(
            user_id=1,
            persona_data=persona_data,
            account_id=account_id,
            learning_profile=learning_profile,
            scheduling_preferences=scheduling_preferences
        )
        
        # Await tasks
        learning_result, workflow_result = await asyncio.gather(learning_task, workflow_task, return_exceptions=True)
        
        if isinstance(learning_result, Exception):
            logger.error(f"Learning analysis failed: {str(learning_result)}")
            raise learning_result
        
        # Process learning result
        parsed_output = learning_result.final_output_as(EnhancedLearningOutput)
        print("\nAgent Response:")
        print("Insights:", parsed_output.insights)
        print("Recommendations:", parsed_output.recommendations)
        print("Improvement Areas:", parsed_output.improvement_areas)
        
        # Process workflow result
        if workflow_result["success"]:
            logger.info("Workflow completed successfully")
            print("\n=== Learning Analysis ===")
            print(f"Insights: {workflow_result['analysis']['insights']}")
            print("\nRole Model Analysis:")
            for rm, analysis in workflow_result['analysis']['role_model_analysis'].items():
                print(f"- {rm}: {analysis}")
            
            print("\n=== Generated Tweets ===")
            for i, tweet_info in enumerate(workflow_result["tweets"], 1):
                print(f"\nTweet {i}:")
                print(f"Content: {tweet_info['content']}")
                print(f"Quality Metrics: {tweet_info['quality_metrics']}")
                print(f"Engagement Predictions: {tweet_info['engagement_predictions']}")
                print(f"Risk Assessment: {tweet_info['risk_assessment']}")
        else:
            logger.error(f"Workflow failed: {workflow_result['error']}")
            print(f"Workflow failed: {workflow_result['error']}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_workflow())