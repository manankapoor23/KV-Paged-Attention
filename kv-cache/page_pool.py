## we have to manage the free pages too so this is for them
## importing our class from page
from .page import KVPage

class PagePool:
    def __init__(self,num_pages,page_size):
        self.page_size = page_size
        self.free_pages = []
        self.used_pages = {}

        for i in range(num_pages):
            self.free_pages.append(KVPage(i,page_size)) ## i is the page id and then we have page_size

    def allocate_page(self):
        if not self.free_pages:
            raise RuntimeError("we are out of pages")
        page = self.free_pages.pop()
        self.used_pages[page.page_id]=page
        print(f"[KVPager] Page fault -> allocated page {page.page_id}") 
        return page
    def free_page(self,page_id):
        page = self.used_pages.pop(page_id)
        page.used=0
        self.free_page.append(page)
        print(f"[KVPager] freed page {page_id}")
