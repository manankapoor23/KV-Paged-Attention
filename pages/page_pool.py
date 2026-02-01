## we have to manage the free pages too so this is for them
## importing our class from page
from .page import KVPage

class PagePool:
    def __init__(self, num_pages, page_size, num_layers, num_heads, head_dim, device):
        self.page_size = page_size
        self.free_pages = [] ## reusable memory
        self.used_pages = {} ## currently alloacted memory 

        for i in range(num_pages):
            self.free_pages.append(KVPage(i, page_size, num_layers, num_heads, head_dim, device)) ## i is the page id and then we have page_size

    def allocate_page(self):
        if not self.free_pages:
            raise RuntimeError("we are out of pages")
        page = self.free_pages.pop()
        self.used_pages[page.page_id]=page
        print(f"[KVPager] Page fault -> allocated page {page.page_id}") 
        return page
    
    def free_page(self,page_id):
        print(f"[KVPager] Freeing page {page.page_id}")
        page = self.used_pages.pop(page_id)
        page.used=0
        self.free_page.append(page)
        print(f"[KVPager] freed page {page_id}")


## memory computer per page and not per sequence as seen in Total size of KV cache in bytes = 2 times batch_size * sequence_length * num_heads * num_layers * num_dimensions * sizeof (FP16)
##That formula describes the size of a contiguous KV cache, but paged KV caching deliberately breaks the contiguous-memory assumption, so we first build per-page KV storage and reintroduce the formula later as a sum over pages.