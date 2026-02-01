"""
Narration engine for explaining KV cache events in human-readable terms.
Maps event types to educational explanations.
"""

from typing import Dict, List, Any


class NarrationEngine:
    """Converts technical events into human-readable explanations"""

    def __init__(self):
        self.token_text = {}  # token_idx -> text token
        self.page_allocations = {}  # page_id -> [token_indices]
        self.event_explanations = {}  # event_id -> explanation

    def set_tokens(self, tokens: List[str]):
        """Store token text for reference"""
        self.token_text = {i: t for i, t in enumerate(tokens)}

    def explain_event(self, event: Dict[str, Any], all_events: List[Dict]) -> str:
        """
        Generate a 2-3 line human-readable explanation for an event.
        
        Args:
            event: The event to explain
            all_events: All events up to this point (for context)
            
        Returns:
            String explanation suitable for UI display
        """
        event_type = event["event_type"]
        details = event["details"]

        if event_type == "page_fault":
            page_id = details.get("page_id")
            return f"📄 A new page (Page {page_id}) was allocated.\nThe current page was full, so the system needed more space."

        elif event_type == "kv_write":
            page_id = details.get("page_id")
            slot = details.get("slot")
            token_idx = details.get("token_idx")
            token_text = self.token_text.get(token_idx, f"Token {token_idx}")
            
            # Track allocation
            if page_id not in self.page_allocations:
                self.page_allocations[page_id] = []
            self.page_allocations[page_id].append(token_idx)
            
            return f"✍️ Writing token '{token_text}' to Page {page_id}, Slot {slot}.\nThis stores the key-value data for this position."

        elif event_type == "copy_on_write":
            source_page = details.get("source_page_id")
            new_page = details.get("new_page_id")
            return f"🔀 Copy-on-Write: Page {source_page} was shared.\nWe copied it to Page {new_page} to prevent data conflicts between requests."

        elif event_type == "prefix_reuse":
            num_pages = details.get("num_pages", 1)
            return f"♻️ Prefix Cache Hit: {num_pages} page(s) were reused.\nThese pages contained tokens we'd already computed—no need to recalculate!"

        elif event_type == "page_freed":
            page_id = details.get("page_id")
            return f"🗑️ Page {page_id} was freed.\nIt's no longer needed, freeing memory for new pages."

        elif event_type == "token_step":
            token_idx = details.get("token_idx")
            token_text = self.token_text.get(token_idx, f"Token {token_idx}")
            return f"📝 Processing token: '{token_text}' (index {token_idx}).\nComputing attention and moving to the next token."

        elif event_type == "decode_start":
            num_pages = details.get("num_pages", 1)
            return f"▶️ Decode phase started.\nWe now have {num_pages} page(s) containing all the keys and values computed so far."

        elif event_type == "decode_end":
            return f"✅ Decode phase complete.\nAll tokens have been processed and stored in the page pool."

        elif event_type == "request_start":
            prompt = details.get("prompt", "")
            return f"🚀 New request started: \"{prompt}\"\nThe system will now allocate and manage KV cache for this text."

        elif event_type == "request_end":
            return f"🏁 Request complete.\nAll KV data for this request has been processed and stored."

        else:
            return f"Event: {event_type}\nDetails: {str(details)[:100]}"

    def get_token_mapping(self, step: int, all_events: List[Dict]) -> Dict[int, Dict]:
        """
        Get a summary of which tokens are on which pages up to this step.
        
        Returns:
            {token_idx: {"text": str, "page_id": int, "slot": int}}
        """
        mapping = {}
        events = all_events[:step + 1]

        for event in events:
            if event["event_type"] == "kv_write":
                token_idx = event["details"].get("token_idx")
                page_id = event["details"].get("page_id")
                slot = event["details"].get("slot")
                token_text = self.token_text.get(token_idx, f"Token {token_idx}")
                
                mapping[token_idx] = {
                    "text": token_text,
                    "page_id": page_id,
                    "slot": slot,
                    "event_id": event.get("event_id"),
                }

        return mapping

    def get_page_summary(self, step: int, all_events: List[Dict]) -> Dict[int, Dict]:
        """
        Get state of all pages up to this step.
        
        Returns:
            {page_id: {"used_slots": int, "ref_count": int, "status": str}}
        """
        pages = {}
        events = all_events[:step + 1]

        for event in events:
            event_type = event["event_type"]
            details = event["details"]

            if event_type == "page_fault":
                page_id = details.get("page_id")
                pages[page_id] = {
                    "used_slots": 0,
                    "ref_count": 1,
                    "status": "allocated",
                    "created_at": event.get("event_id"),
                }

            elif event_type == "kv_write":
                page_id = details.get("page_id")
                if page_id in pages:
                    pages[page_id]["used_slots"] = max(
                        pages[page_id]["used_slots"],
                        details.get("slot", 0) + 1,
                    )

            elif event_type == "copy_on_write":
                source_id = details.get("source_page_id")
                new_id = details.get("new_page_id")
                if source_id in pages:
                    pages[source_id]["ref_count"] -= 1
                    pages[new_id] = {
                        "used_slots": pages[source_id]["used_slots"],
                        "ref_count": 1,
                        "status": "copy",
                        "source": source_id,
                    }

            elif event_type == "page_freed":
                page_id = details.get("page_id")
                if page_id in pages:
                    pages[page_id]["status"] = "freed"

        return pages
