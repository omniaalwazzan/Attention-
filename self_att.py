import torch
import math 
import torch.nn.functional as F



def selfAttention(q, k, v, mask=None):
    d_k = q.size()[-1]
    

    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention#


bs ,h, d_k = 2, 64, 1
q = torch.randn(bs,h, d_k)
k = torch.randn(bs,h, d_k)
v = torch.randn(bs,h, d_k)
values, attention = selfAttention(q, k, v)
print("Q\n", q.shape)
print("K\n", k.shape)
print("V\n", v.shape)
print("Values\n", values.shape)
print("Attention\n", attention)