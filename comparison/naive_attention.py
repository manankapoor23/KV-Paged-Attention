def NaiveAttention(Q,K,V):
    import torch 
    import math
    scores = torch.matmul(K,Q)/math.sqrt(Q.shape[0])
    weights = torch.softmax(scores,dim=0)
    output = torch.sum(weights.unsqueeze(1)*V,dim=0)
    return output