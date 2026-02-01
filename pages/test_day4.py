import torch
from pages.page_pool import PagePool
from pages.page_table import PageTable
from pages.paged_kv_reader import gather_paged_kv
from pages.attention import scaled_dot_product_attention

num_layers = 1
num_heads = 1
head_dim = 4
page_size = 2
num_pages = 3
device = "cpu"
num_tokens = 5

pool = PagePool(num_pages, page_size, num_layers, num_heads, head_dim, device)
page_table = PageTable()
pages = []

for token_idx in range(num_tokens):
    if not pages or not pages[-1].has_space():
        pages.append(pool.allocate_page())
    page = pages[-1]
    slot = page.allocate_slot()
    page.K[0,0,slot]=torch.randn(head_dim)
    page.V[0,0,slot]=torch.randn(head_dim)
    
    page_table.add(page.page_id,slot)

Q=torch.randn(head_dim)
K_seq ,V_seq = gather_paged_kv(pages,page_table,layer_idx=0,head_idx=0)
output = scaled_dot_product_attention(Q,K_seq,V_seq)
print("Paged attention o/p : ",output)

## a page fault happens when the system needs the storage for the token's KV
## this output was computed without a contiguos kv tensor 
## attention logic is it didnt care where kv lived
## paging changes only memory layout , not the attention dynamics
## attention weights depend on Q


