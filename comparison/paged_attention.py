def PagedAttention(Q, pages, page_table, layer_idx, head_idx):
    import torch 
    import math
    K_list =[]
    V_list = []
    
    ## core of paged attention 
    for token_idx in range(len(page_table.table)):
        page_id ,slot = page_table.lookup(token_idx)
        page=pages[page_id]

        K_list.append(page.K[layer_idx,head_idx,slot])
        V_list.append(page.V[layer_idx,head_idx,slot])

    K = torch.stack(K_list,dim=0)
    V=torch.stack(V_list,dim=0)

    ## now comes the math part which is same for naive and paged
    scores = torch.matmul(K,Q)/math.sqrt(Q.shape[0])
    weights = torch.softmax(scores,dim=0)
    output = torch.sum(weights.unsqueeze(1)*V,dim=0)

    return output 


