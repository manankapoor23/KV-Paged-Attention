from pages.page_pool import PagePool
pool = PagePool(num_pages=3,page_size=4)## constructor initialised 

pages = []
for i in range(10):
    if not pages or not pages[-1].has_space():
        pages.append(pool.allocate_page())
    slot = pages[-1].allocate_slot()
    print(f"Token {i} stores in page {pages[-1].page_id},Slot {slot}")