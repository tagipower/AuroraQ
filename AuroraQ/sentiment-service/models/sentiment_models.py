# models/sentiment_models.py
"""Pydantic models for sentiment analysis API"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"


class NewsArticle(BaseModel):
    """News article data model"""
    title: str = Field(..., description="Article title")
    content: Optional[str] = Field(None, description="Article content/summary")
    source: Optional[str] = Field(None, description="News source")
    url: Optional[str] = Field(None, description="Article URL")
    published: Optional[datetime] = Field(None, description="Publication timestamp")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Article keywords")
    engagement: Optional[Dict[str, int]] = Field(default_factory=dict, description="Engagement metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SentimentRequest(BaseModel):
    """Single sentiment analysis request"""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=10000)
    symbol: Optional[str] = Field("CRYPTO", description="Asset symbol for context")
    include_detailed: bool = Field(False, description="Include detailed analysis results")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class SentimentResponse(BaseModel):
    """Single sentiment analysis response"""
    sentiment_score: float = Field(..., description="Sentiment score (0.0 to 1.0)")
    label: SentimentLabel = Field(..., description="Sentiment classification")
    confidence: float = Field(..., description="Analysis confidence (0.0 to 1.0)")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    
    # Optional detailed analysis
    keywords: Optional[List[str]] = Field(None, description="Extracted keywords")
    scenario_tag: Optional[str] = Field(None, description="Market scenario tag")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchSentimentRequest(BaseModel):
    """Batch sentiment analysis request"""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    symbol: Optional[str] = Field("CRYPTO", description="Asset symbol for context")
    include_detailed: bool = Field(False, description="Include detailed analysis results")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        cleaned_texts = []
        for text in v:
            if isinstance(text, str) and text.strip():
                cleaned_texts.append(text.strip())
        
        if not cleaned_texts:
            raise ValueError('At least one valid text required')
        
        return cleaned_texts


class BatchSentimentResponse(BaseModel):
    """Batch sentiment analysis response"""
    results: List[SentimentResponse] = Field(..., description="List of sentiment analysis results")
    total_count: int = Field(..., description="Total number of processed texts")
    average_score: float = Field(..., description="Average sentiment score")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FusionRequest(BaseModel):
    """Multi-source sentiment fusion request"""
    sentiment_scores: Dict[str, float] = Field(
        ..., 
        description="Source-specific sentiment scores",
        example={"news": 0.7, "social": 0.6, "technical": 0.5}
    )
    symbol: Optional[str] = Field("BTCUSDT", description="Trading symbol")
    timestamp: Optional[datetime] = Field(None, description="Analysis timestamp")
    
    @validator('sentiment_scores')
    def validate_scores(cls, v):
        if not v:
            raise ValueError('Sentiment scores cannot be empty')
        
        for source, score in v.items():
            if not isinstance(score, (int, float)):
                raise ValueError(f'Score for {source} must be numeric')
            if not (0.0 <= score <= 1.0):
                raise ValueError(f'Score for {source} must be between 0.0 and 1.0')
        
        return v


class FusionResponse(BaseModel):
    """Multi-source sentiment fusion response"""
    fused_score: float = Field(..., description="Fused sentiment score (0.0 to 1.0)")
    confidence: float = Field(..., description="Fusion confidence (0.0 to 1.0)")
    trend: str = Field(..., description="Market trend indication")
    volatility: float = Field(..., description="Sentiment volatility")
    
    # Fusion details
    raw_scores: Dict[str, float] = Field(..., description="Original input scores")
    weights_used: Dict[str, float] = Field(..., description="Weights applied in fusion")
    sources_count: int = Field(..., description="Number of sources used")
    
    # Metadata
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    fusion_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional fusion metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SentimentResult(BaseModel):
    """Detailed sentiment analysis result"""
    text: str = Field(..., description="Analyzed text")
    sentiment_score: float = Field(..., description="Sentiment score (0.0 to 1.0)")
    label: SentimentLabel = Field(..., description="Sentiment classification")
    confidence: float = Field(..., description="Analysis confidence")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    scenario_tag: str = Field("", description="Market scenario tag")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FusedSentiment(BaseModel):
    """Fused sentiment result with history"""
    timestamp: datetime = Field(..., description="Analysis timestamp")
    symbol: str = Field(..., description="Trading symbol")
    fused_score: float = Field(..., description="Fused sentiment score")
    raw_scores: Dict[str, float] = Field(..., description="Original source scores")
    weights_used: Dict[str, float] = Field(..., description="Applied fusion weights")
    confidence: float = Field(..., description="Fusion confidence")
    volatility: float = Field(..., description="Sentiment volatility")
    trend: str = Field(..., description="Market trend")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    
    # Component health
    components: Dict[str, str] = Field(default_factory=dict, description="Component health status")
    
    # Performance metrics
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Additional utility models
class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    loaded: bool = Field(..., description="Whether model is loaded")
    load_time: Optional[float] = Field(None, description="Model load time in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    

class ServiceStats(BaseModel):
    """Service statistics"""
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time: float = Field(..., description="Average response time in seconds")
    uptime: float = Field(..., description="Service uptime in seconds")
    cache_hit_rate: float = Field(..., description="Cache hit rate as percentage")
    
    # Model statistics
    model_info: Dict[str, ModelInfo] = Field(default_factory=dict, description="Loaded models info")
    
    # Memory and performance
    memory_usage: float = Field(..., description="Current memory usage in MB")
    cpu_usage: float = Field(..., description="Current CPU usage as percentage")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Statistics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }