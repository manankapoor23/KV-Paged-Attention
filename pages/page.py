## file for representing a single kv page that can hold n slots
## fixed size chunk of memory that can store KV entries for a limited nnumber of tokens
import torch
class KVPage:
    def __init__(self,page_id,page_size,num_layers, num_heads, head_dim, device):
        self.page_id=page_id
        self.page_size=page_size
        self.used=0
        self.K = torch.zeros(
            num_layers, num_heads, page_size, head_dim, device=device
        )
        self.V = torch.zeros(
            num_layers, num_heads, page_size, head_dim, device=device
        )

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
