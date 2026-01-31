## virtual memory for tokens
class PageTable:
    def __init__(self):
        self.table=[]
    def add(self,page_id,slot):
        self.table.append((page_id,slot))
    def lookup(self,token_index):
        return self.table[token_index]
