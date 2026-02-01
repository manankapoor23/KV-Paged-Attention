## this is a mapper which maps the prefix's hash to the list of page_ids

class PrefixCache:
    def __init__(self):
        self.cache={}
    def get(self,prefix_key):
        return self.cache.get(prefix_key)
    def put(self,prefix_key,pages):
        self.cache[prefix_key]=pages

