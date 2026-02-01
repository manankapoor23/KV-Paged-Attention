import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pages.page_pool import PagePool
from pages.page_table import PageTable
from pages.prefix_cache import PrefixCache

# =====================================================
# GLOBAL MODEL LOAD (ONCE PER PROCESS)
# =====================================================
device = "cpu"   # change to "mps" later if you want

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()
model.to(device)

print("MODEL LOADED")

# =====================================================
# CONFIG
# =====================================================
config = model.config

num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads
hidden_size = config.hidden_size
head_dim = hidden_size // num_heads
page_size = 4
num_pages = 8

torch.manual_seed(0)

print(f"Model config: layers={num_layers}, heads={num_heads}, hidden_size={hidden_size}, head_dim={head_dim}")

# =====================================================
# REAL PREFIX COMPUTE (REAL KV)
# =====================================================
def compute_prefix(prefix_tokens, page_pool):
    pages = []
    page_table = PageTable()
    current_page = None

    prompt = " ".join(prefix_tokens)
    input_ids = tokenizer(
        prompt, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True
        )

    past_key_values = outputs.past_key_values
    seq_len = input_ids.shape[1]

    for token_idx in range(seq_len):
        if current_page is None or not current_page.has_space():
            current_page = page_pool.allocate_page()
            pages.append(current_page)

        slot = current_page.allocate_slot()

        
        for layer_idx, kv_pair in enumerate(past_key_values):
            K, V = kv_pair[0], kv_pair[1]
            current_page.K[layer_idx, :, slot, :] = K[0, :, token_idx, :]
            current_page.V[layer_idx, :, slot, :] = V[0, :, token_idx, :]

        page_table.add(current_page.page_id, slot)

    return pages, page_table


# =====================================================
# DRIVER
# =====================================================
def main():
    page_pool = PagePool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device
    )

    prefix_cache = PrefixCache()


    # REQUEST 1

    print("\n=== REQUEST 1 ===")
    prefix_tokens = ["You", "are", "a", "helpful","Agent","Who","Is","my","Teacher","of","english"]

    prefix_key = hash(tuple(prefix_tokens))
    cached = prefix_cache.get(prefix_key)

    if cached:
        pages, page_table = cached
        for p in pages:
            p.ref_count += 1
    else:
        pages, page_table = compute_prefix(prefix_tokens, page_pool)
        for p in pages:
            p.ref_count = 1
        prefix_cache.put(prefix_key, (pages, page_table))

    #decode one new token (COW-safe) 
    current_page = pages[-1]

    if current_page.ref_count > 1:
        print("[COW] Request 1")
        new_page = page_pool.allocate_page()
        new_page.K[:] = current_page.K[:]
        new_page.V[:] = current_page.V[:]

        current_page.ref_count -= 1
        new_page.ref_count = 1

        pages[-1] = new_page
        current_page = new_page

    slot = current_page.allocate_slot()

    # NOTE: decode KV is fake here (OK for Day 5)
    current_page.K[0, 0, slot] = torch.randn(head_dim)
    current_page.V[0, 0, slot] = torch.randn(head_dim)
    page_table.add(current_page.page_id, slot)

    print("Request 1 pages:", [p.page_id for p in pages])


    # REQUEST 2 (SAME PREFIX)

    print("\n=== REQUEST 2 ===")
    pages2, page_table2 = prefix_cache.get(prefix_key)

    for p in pages2:
        p.ref_count += 1

    current_page = pages2[-1]

    if current_page.ref_count > 1:
        print("[COW] Request 2")
        new_page = page_pool.allocate_page()
        new_page.K[:] = current_page.K[:]
        new_page.V[:] = current_page.V[:]

        current_page.ref_count -= 1
        new_page.ref_count = 1

        pages2[-1] = new_page
        current_page = new_page

    slot = current_page.allocate_slot()
    current_page.K[0, 0, slot] = torch.randn(head_dim)
    current_page.V[0, 0, slot] = torch.randn(head_dim)
    page_table2.add(current_page.page_id, slot)

    print("Request 2 pages:", [p.page_id for p in pages2])

    # =================================================
    # CLEANUP
    # =================================================
    print("\n=== CLEANUP ===")
    for p in pages:
        p.ref_count -= 1
        if p.ref_count == 0:
            page_pool.free_page(p.page_id)

    for p in pages2:
        p.ref_count -= 1
        if p.ref_count == 0:
            page_pool.free_page(p.page_id)


if __name__ == "__main__":
    main()
