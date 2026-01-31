import torch
from pages.page_pool import PagePool
from pages.page_table import PageTable

# Fake model config (small on purpose)
num_layers = 2
num_heads = 2
head_dim = 4
page_size = 5
num_pages = 3
device = "cpu"

pool = PagePool(
    num_pages, page_size, num_layers, num_heads, head_dim, device
)
page_table = PageTable()
pages = []

for token_idx in range(10):
    if not pages or not pages[-1].has_space():
        pages.append(pool.allocate_page())

    page = pages[-1]
    slot = page.allocate_slot()

    # Fake KV (pretend it came from transformer)
    fake_K = torch.randn(num_layers, num_heads, head_dim)
    fake_V = torch.randn(num_layers, num_heads, head_dim)

    page.K[:, :, slot, :] = fake_K
    page.V[:, :, slot, :] = fake_V

    page_table.add(page.page_id, slot)

    print(
        f"Token {token_idx} → Page {page.page_id}, Slot {slot}"
    )
print(f"USES {len(pages)} pages ")
