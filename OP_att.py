import math 
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from numpy import random


def selfAttention(q, k, v):    
    attn_logits = q.view(q.shape[0], q.shape[1], q.shape[2])  * k.view(k.shape[0], -1,k.shape[1])
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention#


batch_size ,h_dim, model_dim = 10, 4,1
q = torch.randn(batch_size,h_dim, model_dim)
k = torch.randn(batch_size,h_dim, model_dim)
v = torch.randn(batch_size,h_dim, model_dim)
values, attention = selfAttention(q, k, v)

#print("Shape of Q\n", q.shape)
#print("Shape of K\n", k.shape)
#print("Shape of V\n", v.shape)
print("Values\n", values)
print("Attention\n", attention)



#plt.imshow(np.random.random((50,50)))
for i in range(10):
    
    plt.imshow(attention[i,:,:])
    #plt.imshow(attention[1,:,:])
    plt.colorbar()
    plt.show()


##########

Q = torch.Tensor([[1,3]
                 ,[2,4]]).float()[None]

K = torch.Tensor([[1,3],
                  [2,4]]).float()[None]

out_ =Q.view(Q.shape[0], Q.shape[1], -1) + K.view(K.shape[0],-1, K.shape[1])

plt.imshow(out_[0,:,:])
plt.colorbar()
plt.show()

# scores = torch.matmul(Q, K) 
# plt.imshow(scores)
# plt.colorbar()
# plt.show()

