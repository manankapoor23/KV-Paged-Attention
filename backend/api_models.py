"""
Pydantic models for API requests/responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SimulateRequest(BaseModel):
    """Request to run a simulation"""
    prompt: str
    max_tokens: int = 5


class ModelConfig(BaseModel):
    """Model configuration info"""
    layers: int
    heads: int
    hidden_size: int
    head_dim: int


class Event(BaseModel):
    """Event record"""
    event_id: int
    event_type: str
    timestamp: float
    request_id: int
    details: Dict[str, Any]
    narration: Optional[str] = None  # Human-readable explanation
    changed_elements: Optional[List[str]] = None  # Elements that changed this step


class PageState(BaseModel):
    """State of a page at a given time"""
    page_id: int
    used_slots: int
    total_slots: int
    ref_count: int
    is_freed: bool


class SimulateResponse(BaseModel):
    """Response from simulation"""
    model: ModelConfig
    events: List[Event]
    final_pages: List[int]
    page_states: List[PageState]
    summary: Dict[str, Any]
    tokens: List[str]  # Token text for reference
    token_timeline: Optional[Dict[int, Dict]] = None  # Token -> page mappings
