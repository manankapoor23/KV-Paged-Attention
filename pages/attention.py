import torch 
import math


def scaled_dot_product_attention(Q,K,V):
    """
    Q: [head_dim]
    K: [seq_len, head_dim]
    V: [seq_len, head_dim]
    """
    scores = torch.matmul(K,Q)/math.sqrt(Q.shape[0])
    weights = torch.softmax(scores,dim=0)
    output = torch.sum(weights.unsqueeze(1)*V,dim=0)
    return output

