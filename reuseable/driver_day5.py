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


def main():
    page_pool = PagePool(num_pages,num_layers,num_heads,page_size,head_dim)
    prefix_cache = PrefixCache()

    print("\n REQUEST 1 ")
    prefix_tokens = ["You", "are", "a", "helpful"]

    prefix_key = hash(tuple(prefix_tokens))
    cached = prefix_cache.get(prefix_key)

    if cached:
        pages,page_table = cached
        for p in pages:
            p.ref_count_+=1
    else:
        pages,page_table=compute_prefix(prefix_tokens,page_pool)
        for p in pages:
            p.ref_count+=1
        prefix_cache.put(prefix_key,(pages,page_table))
    

    ## continue decoding ( new token )
    current_page = pages[-1]
    if current_page.ref_count>1:
        ## WE DEMAND A COPY ON WRITE NOW 
        
        ## step-1 : make a new page
        new_page = page_pool.allocate_page()
        new_page.K[:]=current_page.K[:]
        new_page.V[:]=current_page.V[:]

        ## decrease the current pages reference counter
        current_page.ref_count -=1
        
        ##assigning
        pages[-1]=new_page
        current_page=new_page
    

    slot = current_page.allocate_slot()
    current_page.K[0,0,slot]=torch.randn(head_dim)
    current_page.V[0,0,slot]=torch.randn(head_dim)
    page_table.add(current_page.page_id,slot)

    print("REQUEST 1 PAGES : ",[p.page_id for p in pages])


    ## REQUEST 2 of the same prefix 

    print("\n REQUEST 2 ")
    cached= prefix_cache.get(prefix_key)
    pages2,page_table2=cached

    for p in pages2:
        p.ref_count+=1
    current_page=pages2[-1]
    if current_page.ref_count>1:
        ## perform copy on write
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

    # ---------------------
    # CLEANUP REQUESTS
    # ---------------------
    print("\n=== CLEANUP ===")
    for p in pages:
        p.ref_count -= 1
        if p.ref_count == 0:
            page_pool.free_page(p)

    for p in pages2:
        p.ref_count -= 1
        if p.ref_count == 0:
            page_pool.free_page(p)


if __name__ == "__main__":
    main()




