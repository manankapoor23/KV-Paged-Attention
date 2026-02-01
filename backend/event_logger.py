"""
Central event logging system for paged KV cache visualization.
Records all system events for replay and visualization.
"""

from dataclasses import dataclass, asdict
from typing import List, Literal
from enum import Enum


class EventType(str, Enum):
    PAGE_FAULT = "page_fault"
    KV_WRITE = "kv_write"
    PREFIX_REUSE = "prefix_reuse"
    COW = "copy_on_write"
    PAGE_FREED = "page_freed"
    TOKEN_STEP = "token_step"
    DECODE_START = "decode_start"
    DECODE_END = "decode_end"
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"


@dataclass
class Event:
    """Single event in the system"""
    event_id: int
    event_type: EventType
    timestamp: float
    request_id: int
    details: dict

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "details": self.details,
        }


class EventLogger:
    """
    Logs all events from the paged KV system.
    Used for visualization and replay.
    """

    def __init__(self):
        self.events: List[Event] = []
        self.event_counter = 0
        self.request_counter = 0
        self.current_request_id = 0
        self.start_time = None
        
    def start_request(self, prompt: str):
        """Mark the start of a new request"""
        import time
        if self.start_time is None:
            self.start_time = time.time()
        
        self.request_counter += 1
        self.current_request_id = self.request_counter
        
        self.log_event(
            EventType.REQUEST_START,
            {"prompt": prompt, "request_id": self.current_request_id}
        )
        
    def end_request(self):
        """Mark the end of current request"""
        self.log_event(
            EventType.REQUEST_END,
            {"request_id": self.current_request_id}
        )

    def log_page_fault(self, page_id: int):
        """Log page allocation (page fault)"""
        self.log_event(
            EventType.PAGE_FAULT,
            {"page_id": page_id}
        )

    def log_kv_write(self, page_id: int, slot: int, token_idx: int, layer: int):
        """Log KV write to a page slot"""
        self.log_event(
            EventType.KV_WRITE,
            {
                "page_id": page_id,
                "slot": slot,
                "token_idx": token_idx,
                "layer": layer,
            }
        )

    def log_prefix_reuse(self, prefix_key: str, num_pages: int):
        """Log prefix cache hit"""
        self.log_event(
            EventType.PREFIX_REUSE,
            {"prefix_key": prefix_key, "num_pages": num_pages}
        )

    def log_copy_on_write(self, source_page_id: int, new_page_id: int):
        """Log copy-on-write event"""
        self.log_event(
            EventType.COW,
            {"source_page_id": source_page_id, "new_page_id": new_page_id}
        )

    def log_page_freed(self, page_id: int):
        """Log page freeing"""
        self.log_event(
            EventType.PAGE_FREED,
            {"page_id": page_id}
        )

    def log_token_step(self, token_idx: int):
        """Log token processing step"""
        self.log_event(
            EventType.TOKEN_STEP,
            {"token_idx": token_idx}
        )

    def log_decode_start(self, num_pages: int):
        """Log start of decode phase"""
        self.log_event(
            EventType.DECODE_START,
            {"num_pages": num_pages}
        )

    def log_decode_end(self):
        """Log end of decode phase"""
        self.log_event(
            EventType.DECODE_END,
            {}
        )

    def log_event(self, event_type: EventType, details: dict):
        """Log a generic event"""
        import time
        
        self.event_counter += 1
        event = Event(
            event_id=self.event_counter,
            event_type=event_type,
            timestamp=time.time() - (self.start_time or time.time()),
            request_id=self.current_request_id,
            details=details,
        )
        self.events.append(event)

    def get_events(self) -> List[dict]:
        """Get all events as dictionaries"""
        return [e.to_dict() for e in self.events]

    def clear(self):
        """Clear all events"""
        self.events.clear()
        self.event_counter = 0

    def get_summary(self) -> dict:
        """Get summary statistics"""
        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "num_requests": self.request_counter,
        }
