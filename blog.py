import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
from agents import Agent, Runner, Tool, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .database import get_db, Keyword

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
PLAGIARISM_API_KEY = os.getenv("PLAGIARISM_API_KEY")

# Validate API keys
if not all([OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, PLAGIARISM_API_KEY]):
    raise ValueError("One or more required API keys are missing in the .env file.")

@dataclass
class BlogConfig:
    keyword: str
    word_count: int = 2000

class KeywordOutput(BaseModel):
    keywords: List[Dict[str, str]]
    primary_keyword: str

class ContentOutput(BaseModel):
    content: List[Dict[str, str]]
    summary: str

class BlogDraftOutput(BaseModel):
    content: str
    meta_title: str
    meta_description: str
    faq: List[Dict[str, str]]

class ValidationOutput(BaseModel):
    is_valid: bool
    plagiarism_score: float
    seo_score: float

class PlagiarismCheckTool(Tool):
    """Tool for checking content plagiarism"""
    
    def __init__(self):
        self.api_key = PLAGIARISM_API_KEY
        self.base_url = "https://api.plagiarism-checker.com/v1/check"  # Replace with actual API endpoint
    
    async def check_plagiarism(self, content: str) -> Dict:
        """Check content for plagiarism"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": content,
                "language": "en",
                "check_type": "web",  # Check against web content
                "sensitivity": "high"  # High sensitivity for strict checking
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "score": result.get("plagiarism_score", 0.0),
                "sources": result.get("matched_sources", []),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error checking plagiarism: {e}")
            return {
                "score": 0.0,
                "sources": [],
                "status": "error",
                "error": str(e)
            }

class GoogleCSETool(Tool):
    """Tool for searching keywords using Google Programmable Search Engine"""
    
    def __init__(self):
        self.api_key = GOOGLE_API_KEY
        self.cse_id = GOOGLE_CSE_ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search_keywords(self, query: str) -> List[Dict]:
        """Search for keywords using Google CSE"""
        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 10  # Number of results to return
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract keywords from search results
            keywords = []
            for item in data.get("items", []):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                
                # Add the full title as a keyword
                keywords.append({
                    "keyword": title,
                    "search_volume": 1000,  # Placeholder value
                    "competition": 0.5,     # Placeholder value
                    "cpc": 1.0,            # Placeholder value
                    "source": "google_cse"
                })
                
                # Extract key phrases from title and snippet
                key_phrases = self._extract_key_phrases(title + " " + snippet)
                for phrase in key_phrases:
                    keywords.append({
                        "keyword": phrase,
                        "search_volume": 800,  # Placeholder value
                        "competition": 0.4,    # Placeholder value
                        "cpc": 0.8,           # Placeholder value
                        "source": "google_cse"
                    })
            
            return keywords
            
        except Exception as e:
            print(f"Error searching Google CSE: {e}")
            return []
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract meaningful key phrases from text"""
        words = text.lower().split()
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = [w for w in words if w not in common_words and len(w) > 2]
        
        phrases = []
        for i in range(len(words) - 1):
            # 2-word phrases
            phrase = f"{words[i]} {words[i+1]}"
            phrases.append(phrase)
            
            # 3-word phrases
            if i < len(words) - 2:
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(phrase)
        
        return phrases

# Define the Keyword Research Agent with Google CSE tool
keyword_agent = Agent(
    name="Keyword Research Agent",
    instructions="""
    You are an expert in keyword research and SEO. Your task is to:
    1. Analyze the given topic
    2. Search for keywords in the CMS database
    3. Use Google CSE to find additional relevant keywords
    4. Analyze keyword performance metrics
    5. Select the most promising primary keyword
    
    Return a structured list of keywords with metrics from both CMS and Google CSE.
    """,
    output_type=KeywordOutput,
    tools=[GoogleCSETool()]
)

# Define the Content Research Agent
content_agent = Agent(
    name="Content Research Agent",
    instructions="""
    You are a content research specialist. Your task is to:
    1. Search for top content related to the keyword
    2. Analyze and summarize the content
    3. Identify key points and unique angles
    4. Extract relevant information for blog writing
    
    Return a summary of findings and key content points.
    """,
    output_type=ContentOutput
)

# Define the Blog Writing Agent
blog_writer_agent = Agent(
    name="Blog Writer Agent",
    instructions="""
    You are an expert blog writer. Your task is to:
    1. Create engaging, informative blog content
    2. Write SEO-optimized meta title and description
    3. Include relevant FAQ sections
    4. Ensure proper structure and formatting
    5. The blog must contain at least 2000 characters
    
    Return a complete blog post with all required elements.
    """,
    output_type=BlogDraftOutput
)

# Define the Validation Agent with plagiarism checking tool
validation_agent = Agent(
    name="Validation Agent",
    instructions="""
    You are a content quality specialist. Your task is to:
    1. Check SEO optimization
    2. Use the plagiarism checker to verify content uniqueness
    3. Ensure proper structure
    4. Validate readability and engagement
    5. Make sure plagiarism score is below 5%
    
    Return validation results with scores.
    """,
    output_type=ValidationOutput,
    tools=[PlagiarismCheckTool()]
)

class BlogCreator:
    def __init__(self):
        # Initialize agents with proper handoffs
        self.keyword_agent = keyword_agent
        self.content_agent = content_agent
        self.blog_writer_agent = blog_writer_agent
        self.validation_agent = validation_agent
        
        # Set up handoffs between agents
        self.keyword_agent.handoffs = [self.content_agent]
        self.content_agent.handoffs = [self.blog_writer_agent]
        self.blog_writer_agent.handoffs = [self.validation_agent]
        
        # Initialize tools
        self.db = next(get_db())
        self.google_cse_tool = GoogleCSETool()
        self.plagiarism_tool = PlagiarismCheckTool()
        
        # Add input guardrails
        self._setup_guardrails()

    def _setup_guardrails(self):
        """Set up input guardrails for validation"""
        async def topic_guardrail(ctx, agent, input_data):
            if not isinstance(input_data, str) or len(input_data.strip()) < 3:
                return GuardrailFunctionOutput(
                    output_info={"error": "Invalid topic"},
                    tripwire_triggered=True
                )
            return GuardrailFunctionOutput(
                output_info={"valid": True},
                tripwire_triggered=False
            )

        async def word_count_guardrail(ctx, agent, input_data):
            if not isinstance(input_data, int) or input_data < 2000:
                return GuardrailFunctionOutput(
                    output_info={"error": "Word count must be at least 2000 characters"},
                    tripwire_triggered=True
                )
            return GuardrailFunctionOutput(
                output_info={"valid": True},
                tripwire_triggered=False
            )

        async def content_length_guardrail(ctx, agent, input_data):
            if not isinstance(input_data, str) or len(input_data.strip()) < 2000:
                return GuardrailFunctionOutput(
                    output_info={"error": "Blog content must be at least 2000 characters"},
                    tripwire_triggered=True
                )
            return GuardrailFunctionOutput(
                output_info={"valid": True},
                tripwire_triggered=False
            )

        # Add guardrails to agents
        self.keyword_agent.input_guardrails = [
            InputGuardrail(guardrail_function=topic_guardrail)
        ]
        self.blog_writer_agent.input_guardrails = [
            InputGuardrail(guardrail_function=word_count_guardrail),
            InputGuardrail(guardrail_function=content_length_guardrail)
        ]

    async def search_keywords_in_cms(self, topic: str) -> List[Dict]:
        """Search for keywords in the CMS database"""
        keywords = self.db.query(Keyword).filter(
            Keyword.keyword.ilike(f"%{topic}%")
        ).all()
        
        return [
            {
                "keyword": kw.keyword,
                "search_volume": kw.search_volume,
                "competition": kw.competition,
                "cpc": kw.cpc,
                "difficulty": kw.difficulty,
                "trend": kw.trend,
                "source": "cms"
            }
            for kw in keywords
        ]

    async def create_blog(self, topic: str, word_count: int = 2000) -> Dict:
        try:
            # Step 1: Keyword Research using both CMS and Google CSE
            cms_keywords = await self.search_keywords_in_cms(topic)
            google_keywords = await self.google_cse_tool.search_keywords(topic)
            
            all_keywords = cms_keywords + google_keywords
            
            # Use Runner with context for better tracing
            ...