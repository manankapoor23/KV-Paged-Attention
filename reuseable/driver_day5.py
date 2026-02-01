import torch 
import math
from pages.page_pool import PagePool
from pages.page_table import PageTable
from pages.prefix_cache import PrefixCache

num_layers = 1
num_heads = 1
head_dim = 4
page_size = 2
num_pages = 8

torch.manual_seed(0)

def compute_prefix(prefix_tokens,page_pool):
    pages = []
    page_table = PageTable()
    current_page = None

    for _ in prefix_tokens:
        if current_page is None or not current_page.has_space():
            current_page=page_pool.allocate_page()
            pages.append(current_page)
        slot = current_page.allocate_slot()


    ## KV From the model goes here :


    #####
        page_table.add(current_page.page_id,slot)
    return pages,page_table


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