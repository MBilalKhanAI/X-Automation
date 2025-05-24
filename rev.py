from agents import Agent, Runner, set_default_openai_key, Tool
import asyncio
import os
from typing import Dict, Any, List
from datetime import datetime, timedelta
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_default_openai_key(os.getenv("OPENAI_API_KEY"))

# Tool definitions
async def generate_persona_embedding(tone_description: str, hashtags: List[str], style_preferences: str) -> List[float]:
    """Convert persona into vector embedding for AI context use."""
    text = f"{tone_description} {' '.join(hashtags)} {style_preferences}"
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

async def store_persona_embedding(persona_id: str, embedding: List[float]) -> None:
    """Store persona embedding in Supabase."""
    supabase.table("persona_embeddings").insert({
        "persona_id": persona_id,
        "embedding": embedding
    }).execute()

async def query_trends(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Query similar trends based on persona embedding."""
    query = f"""
    SELECT id, metadata, 1 - (embedding <=> '{embedding}') as similarity
    FROM persona_embeddings
    ORDER BY similarity DESC
    LIMIT {top_k}
    """
    result = supabase.rpc("execute_sql", {"query": query}).execute()
    return [{"id": row["id"], "score": row["similarity"], "metadata": row["metadata"]} for row in result.data]

async def generate_tweets(persona_id: str, trend_context: List[str], count: int = 5) -> List[Dict[str, Any]]:
    """Generate tweets based on persona and trend context."""
    prompt = f"Generate {count} professional tweets incorporating these trends: {', '.join(trend_context)}"
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return [{"content": tweet.strip(), "trend_context": trend_context} for tweet in response.choices[0].message.content.split("\n")]

async def assess_risk(tweet: str) -> Dict[str, Any]:
    """Score tweet for backlash risk with cultural sensitivity."""
    prompt = f"Score this tweet for backlash risk (0-100) with cultural sensitivity: {tweet}"
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return eval(response.choices[0].message.content)

async def analyze_competitors(competitor_ids: List[str]) -> Dict[str, Any]:
    """Analyze competitor tweets for high-ranking factors."""
    # Fetch competitor metrics from Supabase
    result = supabase.table("competitor_metrics").select("*").in_("competitor_name", competitor_ids).execute()
    competitor_data = result.data
    
    prompt = f"Analyze these competitors' tweets for engagement factors: {', '.join(competitor_ids)}"
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_tokens=1000
    )
    return eval(response.choices[0].message.content)

# Define tools
tools = [
    Tool(
        name="generate_persona_embedding",
        func=generate_persona_embedding,
        description="Convert persona into vector embedding for AI context use"
    ),
    Tool(
        name="store_persona_embedding",
        func=store_persona_embedding,
        description="Store persona embedding in Supabase"
    ),
    Tool(
        name="query_trends",
        func=query_trends,
        description="Query similar trends based on persona embedding"
    ),
    Tool(
        name="generate_tweets",
        func=generate_tweets,
        description="Generate tweets based on persona and trend context"
    ),
    Tool(
        name="assess_risk",
        func=assess_risk,
        description="Score tweet for backlash risk with cultural sensitivity"
    ),
    Tool(
        name="analyze_competitors",
        func=analyze_competitors,
        description="Analyze competitor tweets for high-ranking factors"
    )
]

# Define agents with tools
tweet_generation_agent = Agent(
    name="TweetGenerationImpactAgent",
    instructions="Act as a professional social media strategist. Generate 5 tweets (max 280 chars) in a professional tone, incorporating hashtags and reply templates. Predict community impact (engagement, sentiment) using historical metrics. Output JSON: {'tweets': [], 'impact': {}}",
    tools=[tools[3], tools[2]]
)

risk_assessment_agent = Agent(
    name="RiskAssessmentAgent",
    instructions="Score 5 tweets for backlash risk (0–100) with cultural sensitivity. Flag crisis-prone posts. Output structured JSON: {'score': int, 'reasons': [], 'crisis_flag': bool}",
    tools=[tools[4]]
)

scheduling_post_analysis_agent = Agent(
    name="SchedulingPostAnalysisAgent",
    instructions="Schedule tweets with formal metadata. Monitor comments every 24 hours for crisis detection. After 24 hours, analyze backlash and trigger alerts for negative spikes (>20% negative sentiment). Output JSON: {'schedule': {}, 'metrics': {}, 'alerts': []}",
    tools=[tools[4]]
)

competitor_analysis_agent = Agent(
    name="CompetitorAnalysisAgent",
    instructions="Analyze competitor tweets for high-ranking factors (engagement, hashtags, timing). Generate formal reports in JSON: {'metrics': [], 'insights': []}",
    tools=[tools[5]]
)

trend_strategy_agent = Agent(
    name="TrendStrategyAgent",
    instructions="Analyze current trends and generate 3+ formal strategy suggestions. Output JSON: {'trends': [], 'suggestions': []}",
    tools=[tools[2], tools[0]]
)

learning_interaction_agent = Agent(
    name="LearningInteractionAgent",
    instructions="Configure client-specific learning profiles and generate formal comment replies. Output JSON: {'profiles': {}, 'replies': [], 'logs': []}",
    tools=[tools[0], tools[1]]
)

async def orchestrate_workflow(user_id: int, persona_data: Dict[str, Any], account_id: str, competitor_ids: List[str], learning_profile: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "tweets": [],
        "impact_analysis": {},
        "risk_scores": [],
        "schedule": {},
        "backlash_results": [],
        "competitor_insights": {},
        "trends": [],
        "strategy_suggestions": [],
        "replies": [],
        "activity_logs": []
    }

    # Step 1: Configure learning profile
    learning_result = await Runner.run(learning_interaction_agent, {"learning_profile": learning_profile})
    result["replies"] = learning_result.final_output.get("replies", [])
    result["activity_logs"] = learning_result.final_output.get("logs", [])

    # Step 2: Generate strategies
    trend_data = await Runner.run(trend_strategy_agent, {"persona_data": persona_data})
    result["trends"] = trend_data.final_output.get("trends", [])
    result["strategy_suggestions"] = trend_data.final_output.get("suggestions", [])

    # Step 3: Generate tweets
    tweet_data = {
        "persona": persona_data["tone"],
        "character_details": persona_data["style_preferences"],
        "hashtags": persona_data["hashtags"],
        "trend_context": result["trends"],
        "account_id": account_id
    }
    tweet_result = await Runner.run(tweet_generation_agent, tweet_data)
    result["tweets"] = tweet_result.final_output.get("tweets", [])
    result["impact_analysis"] = tweet_result.final_output.get("impact", {})

    # Step 4: Assess risks
    scored_tweets = []
    for tweet in result["tweets"]:
        risk_result = await Runner.run(risk_assessment_agent, {"tweet": tweet})
        risk_data = risk_result.final_output
        if risk_data.get("score", 100) < 50:
            scored_tweets.append({
                "tweet": tweet,
                "risk_score": risk_data.get("score", 0),
                "reason": risk_data.get("reasons", []),
                "crisis_flag": risk_data.get("crisis_flag", False)
            })
        result["risk_scores"].append(risk_data)

    # Step 5: Schedule and analyze
    post_ids = []
    for scored_tweet in scored_tweets:
        post_id = str(uuid.uuid4())
        schedule_time = datetime.now() + timedelta(hours=1)
        schedule_result = await Runner.run(scheduling_post_analysis_agent, {
            "post_id": post_id,
            "content": scored_tweet["tweet"],
            "scheduled_time": schedule_time.isoformat()
        })
        post_ids.append(post_id)
        result["schedule"][post_id] = schedule_result.final_output

    # Step 6: Analyze competitors
    competitor_insights = await Runner.run(competitor_analysis_agent, {"competitor_ids": competitor_ids})
    result["competitor_insights"] = competitor_insights.final_output

    # Step 7: Generate replies
    reply_result = await Runner.run(learning_interaction_agent, {
        "comments": ["Sample comment 1", "Sample comment 2"],
        "post_id": post_ids[0] if post_ids else None
    })
    result["replies"] = reply_result.final_output.get("replies", [])

    return result

# Run workflow
if __name__ == "__main__":
    persona_data = {
        "tone": "professional",
        "style_preferences": "concise, polite, innovative",
        "hashtags": ["#Tech", "#Innovation", "#JapanBusiness"]
    }
    account_id = "1"
    competitor_ids = ["comp1", "comp2", "comp3"]
    learning_profile = {
        "role_models": ["TechCorp", "InnovateJP"],
        "industry_info": "Technology, SaaS",
        "competitor_profiles": ["comp1", "comp2"]
    }
    result = asyncio.run(orchestrate_workflow(
        user_id=1,
        persona_data=persona_data,
        account_id=account_id,
        competitor_ids=competitor_ids,
        learning_profile=learning_profile
    ))
    print(result)