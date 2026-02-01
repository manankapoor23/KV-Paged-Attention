"""
Wrapper around driver_day5 logic with event logging.
This is the instrumentation layer - NO core logic changes.
"""

import torch
from typing import Tuple, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.event_logger import EventLogger
from pages.page_pool import PagePool
from pages.page_table import PageTable
from pages.prefix_cache import PrefixCache


class InstrumentedPagePool(PagePool):
    """PagePool wrapper that logs events"""

    def __init__(self, logger: EventLogger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger

    def allocate_page(self):
        page = super().allocate_page()
        self.logger.log_page_fault(page.page_id)
        return page

    def free_page(self, page_id):
        """Free a page by ID (adapted for our wrapper)"""
        # Get the page from used_pages
        page = self.used_pages.pop(page_id, None)
        if page:
            self.logger.log_page_freed(page_id)
            page.used = 0
            page.ref_count = 0
            self.free_pages.append(page)


class KVCacheSimulator:
    """
    Wraps the paged KV cache system with event logging.
    Does NOT modify core logic - only instruments it.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = EventLogger()

        # Load model once
        MODEL_NAME = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        self.model.eval()
        self.model.to(device)

        # Get config
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

    def compute_prefix(
        self, prefix_tokens: List[str], page_pool: InstrumentedPagePool
    ) -> Tuple[List, PageTable]:
        """Compute KV for a prefix - logs all KV writes"""
        pages = []
        page_table = PageTable()
        current_page = None

        prompt = " ".join(prefix_tokens)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)

        past_key_values = outputs.past_key_values
        seq_len = input_ids.shape[1]

        for token_idx in range(seq_len):
            self.logger.log_token_step(token_idx)

            if current_page is None or not current_page.has_space():
                current_page = page_pool.allocate_page()
                pages.append(current_page)

            slot = current_page.allocate_slot()

            for layer_idx, kv_pair in enumerate(past_key_values):
                K, V = kv_pair[0], kv_pair[1]
                current_page.K[layer_idx, :, slot, :] = K[0, :, token_idx, :]
                current_page.V[layer_idx, :, slot, :] = V[0, :, token_idx, :]
                self.logger.log_kv_write(current_page.page_id, slot, token_idx, layer_idx)

            page_table.add(current_page.page_id, slot)

        return pages, page_table

    def simulate(self, prompt: str) -> Dict[str, Any]:
        """Run full simulation with two requests to show prefix reuse"""
        self.logger.clear()

        # Initialize paged system
        page_pool = InstrumentedPagePool(
            self.logger,
            num_pages=8,
            page_size=16,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            device=self.device,
        )

        prefix_cache = PrefixCache()

        # Tokenize prompt to get token text
        prefix_tokens = prompt.split()

        # ===== REQUEST 1 =====
        self.logger.start_request(prompt)

        prefix_key = hash(tuple(prefix_tokens))
        cached = prefix_cache.get(prefix_key)

        if cached:
            pages, page_table = cached
            self.logger.log_prefix_reuse(str(prefix_key), len(pages))
            for p in pages:
                p.ref_count += 1
        else:
            pages, page_table = self.compute_prefix(prefix_tokens, page_pool)
            for p in pages:
                p.ref_count = 1
            prefix_cache.put(prefix_key, (pages, page_table))

        # Decode one new token with COW
        self.logger.log_decode_start(len(pages))
        current_page = pages[-1]

        if current_page.ref_count > 1:
            new_page = page_pool.allocate_page()
            new_page.K[:] = current_page.K[:]
            new_page.V[:] = current_page.V[:]

            current_page.ref_count -= 1
            new_page.ref_count = 1

            self.logger.log_copy_on_write(current_page.page_id, new_page.page_id)

            pages[-1] = new_page
            current_page = new_page

        slot = current_page.allocate_slot()
        current_page.K[0, 0, slot] = torch.randn(self.head_dim)
        current_page.V[0, 0, slot] = torch.randn(self.head_dim)
        page_table.add(current_page.page_id, slot)

        self.logger.log_decode_end()
        self.logger.end_request()

        # ===== REQUEST 2 (same prefix) =====
        self.logger.start_request(prompt)
        pages2, page_table2 = prefix_cache.get(prefix_key)
        self.logger.log_prefix_reuse(str(prefix_key), len(pages2))

        for p in pages2:
            p.ref_count += 1

        current_page = pages2[-1]

        if current_page.ref_count > 1:
            new_page = page_pool.allocate_page()
            new_page.K[:] = current_page.K[:]
            new_page.V[:] = current_page.V[:]

            current_page.ref_count -= 1
            new_page.ref_count = 1

            self.logger.log_copy_on_write(current_page.page_id, new_page.page_id)

            pages2[-1] = new_page
            current_page = new_page

        slot = current_page.allocate_slot()
        current_page.K[0, 0, slot] = torch.randn(self.head_dim)
        current_page.V[0, 0, slot] = torch.randn(self.head_dim)
        page_table2.add(current_page.page_id, slot)

        self.logger.end_request()

        # ===== CLEANUP =====
        for p in pages:
            p.ref_count -= 1
            if p.ref_count == 0:
                page_pool.free_page(p.page_id)

        for p in pages2:
            p.ref_count -= 1
            if p.ref_count == 0:
                page_pool.free_page(p.page_id)

        # Collect final page states
        page_states = [
            {
                "page_id": page.page_id,
                "used_slots": page.used,
                "total_slots": page.page_size,
                "ref_count": page.ref_count,
                "is_freed": page.page_id not in page_pool.used_pages,
            }
            for page in page_pool.free_pages + list(page_pool.used_pages.values())
        ]

        return {
            "model": {
                "layers": self.num_layers,
                "heads": self.num_heads,
                "hidden_size": self.hidden_size,
                "head_dim": self.head_dim,
            },
            "events": self.logger.get_events(),
            "final_pages": [p.page_id for p in pages + pages2],
            "page_states": page_states,
            "summary": self.logger.get_summary(),
            "tokens": prefix_tokens,  # Return token text for visualization
        }
