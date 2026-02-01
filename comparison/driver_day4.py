import torch
import math

# =========================
# CONFIG (small & readable)
# =========================
num_layers = 1
num_heads = 2
head_dim = 4
page_size = 4
num_pages = 4
num_tokens = 16
device = "cpu"

torch.manual_seed(42)

# =========================
# KV PAGE
# =========================
class KVPage:
    def __init__(self, page_id):
        self.page_id = page_id
        self.used = 0
        self.K = torch.zeros(num_layers, num_heads, page_size, head_dim)
        self.V = torch.zeros(num_layers, num_heads, page_size, head_dim)

    def has_space(self):
        return self.used < page_size

    def allocate_slot(self):
        slot = self.used
        self.used += 1
        return slot


# =========================
# PAGE POOL
# =========================
class PagePool:
    def __init__(self):
        self.free_pages = [KVPage(i) for i in range(num_pages)]
        self.used_pages = {}

    def allocate_page(self):
        if not self.free_pages:
            raise RuntimeError("OOM: No KV pages left")

        page = self.free_pages.pop()
        self.used_pages[page.page_id] = page
        print(f"[KVPager] Page fault → allocated page {page.page_id}")
        return page


# =========================
# PAGE TABLE
# =========================
class PageTable:
    def __init__(self):
        self.table = []

    def add(self, page_id, slot):
        self.table.append((page_id, slot))

    def lookup(self, token_idx):
        return self.table[token_idx]


# =========================
# ATTENTION (NAIVE)
# =========================
def NaiveAttention(Q, K, V):
    scores = torch.matmul(K, Q) / math.sqrt(Q.shape[-1])
    weights = torch.softmax(scores, dim=0)
    return torch.sum(weights.unsqueeze(1) * V, dim=0)


# =========================
# ATTENTION (PAGED)
# =========================
def PagedAttention(Q, pages, page_table, layer_idx, head_idx):
    K_list, V_list = [], []

    for t in range(len(page_table.table)):
        pid, slot = page_table.lookup(t)
        page = pages[pid]
        K_list.append(page.K[layer_idx, head_idx, slot])
        V_list.append(page.V[layer_idx, head_idx, slot])

    K = torch.stack(K_list, dim=0)
    V = torch.stack(V_list, dim=0)

    return NaiveAttention(Q, K, V)


# =========================
# MULTI-HEAD PAGED ATTENTION
# =========================
def multi_head_paged_attention(Q, pages, page_table, layer_idx):
    outputs = []

    for h in range(num_heads):
        out = PagedAttention(Q[h], pages, page_table, layer_idx, h)
        outputs.append(out)

    return torch.stack(outputs, dim=0)


# =========================
# DRIVER
# =========================
def main():
    pool = PagePool()
    page_table = PageTable()
    pages = {}

    current_page = None

    print("\n=== Writing KV into pages ===")
    for token_idx in range(num_tokens):
        if current_page is None or not current_page.has_space():
            current_page = pool.allocate_page()
            pages[current_page.page_id] = current_page

        slot = current_page.allocate_slot()

        # Fake KV from "model"
        K = torch.randn(num_layers, num_heads, head_dim)
        V = torch.randn(num_layers, num_heads, head_dim)

        current_page.K[:, :, slot] = K
        current_page.V[:, :, slot] = V

        page_table.add(current_page.page_id, slot)

        print(f"Token {token_idx} → Page {current_page.page_id}, Slot {slot}")

    # =========================
    # BUILD CONTIGUOUS KV (GROUND TRUTH)
    # =========================
    print("\n=== Building contiguous KV (naive) ===")
    K_naive = torch.stack([
        pages[pid].K[0, :, slot]
        for (pid, slot) in page_table.table
    ], dim=0)  # [seq_len, num_heads, head_dim]

    V_naive = torch.stack([
        pages[pid].V[0, :, slot]
        for (pid, slot) in page_table.table
    ], dim=0)

    # =========================
    # QUERY
    # =========================
    Q = torch.randn(num_heads, head_dim)

    # =========================
    # NAIVE MULTI-HEAD
    # =========================
    print("\n=== Naive multi-head attention ===")
    naive_out = []
    for h in range(num_heads):
        out = NaiveAttention(Q[h], K_naive[:, h], V_naive[:, h])
        naive_out.append(out)
    naive_out = torch.stack(naive_out, dim=0)

    # =========================
    # PAGED MULTI-HEAD
    # =========================
    print("\n=== Paged multi-head attention ===")
    paged_out = multi_head_paged_attention(Q, pages, page_table, layer_idx=0)

    # =========================
    # COMPARISON
    # =========================
    print("\n=== COMPARISON ===")
    print("Naive output:\n", naive_out)
    print("Paged output:\n", paged_out)
    print("Absolute difference:\n", torch.abs(naive_out - paged_out))


if __name__ == "__main__":
    main()
