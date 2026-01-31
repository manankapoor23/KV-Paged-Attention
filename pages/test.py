from pages.page_pool import PagePool
pool = PagePool(num_pages=3,page_size=4)## constructor initialised 

pages = []
for i in range(10):
    if not pages or not pages[-1].has_space(): ## current page full , then allocate
        pages.append(pool.allocate_page())
    slot = pages[-1].allocate_slot()
    print(f"Token {i} stored in page {pages[-1].page_id},Slot {slot}")

## keep in mind , the [-1] here is bcs of how tokens are generated sequentially in a decoder only model 
## pages[-1] means the current page of the sequence 
## We write to the last page because decoding is append-only.
## we have to make the pages store the actual KV tensors now 
