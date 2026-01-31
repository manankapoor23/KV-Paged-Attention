## file for representing a single kv page 
class KVPage:
    def __init__(self,page_id,page_size):
        self.page_id=page_id
        self.page_size=page_size
        self.used=0

    def has_space(self):
        ## checks if the current page has some space 
        return self.used<self.page_size

    ## now we have a page system ready , we have to now allocate something to it
    def allocate_slot(self):
        if not self.has_space():
            raise RuntimeError("The Page is Full")
        slot = self.used ## basically 0 rn
        self.used+=1
        return slot 
## this code models the memory right now 
## these are the properties a page will hold
