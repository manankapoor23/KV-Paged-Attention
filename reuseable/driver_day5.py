import torch 
import math
from pages.page_pool import PagePool
from pages.page_table import PageTable
from pages.paged_kv_reader import gather_paged_kv
from pages.attention import scaled_dot_product_attention
from pages.prefix_cache import PrefixCache

prefix_pages = PrefixCache.get(prefix_key)

if prefix_pages:
    ## this is the reuse page
    for p in prefix_pages:
        p.ref_count+=1
    pages=prefix_pages
else:
    ## compute the prefix
    pages = compute_prefix_kv()
    for p in pages:
        p.ref_count=1
    PrefixCache.put(prefix_key,pages)