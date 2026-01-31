import torch 

def gather_paged_kv(pages,page_table,layer_idx,head_idx):
    '''
    this returns :
        K_seq: [seq_len,head_dim]
        V_seq: [seq_len,head_dim]
    '''
    K_list = []
    V_list = []

    for token_idx in range(len(page_table.table)):
        page_id,slot = page_table.lookup(token_idx)
        page = pages[page_id]

        K_list.append(page.K[layer_idx,head_idx,slot])
        V_list.append(page.V[layer_idx,head_idx,slot])
    K_seq=torch.stack(K_list,dim=0)
    V_seq=torch.stack(V_list,dim=0)

    return K_seq,V_seq

## this is where the logical order is stored
