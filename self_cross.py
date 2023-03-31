# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:29:27 2023

@author: Omnia
"""

import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SelfAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = torch.nn.Linear(d_model, d_model, bias=False)
        self.key = torch.nn.Linear(d_model, d_model, bias=False)
        self.value = torch.nn.Linear(d_model, d_model, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, x):
        Q = self.query(x)
        #print("shape of the query",Q.shape)
        K = self.key(x)
        V = self.value(x)
        #V = torch.unsqueeze(V, 2)
        #print("shape of the valu",V.shape)

        # Calculate scaled dot product attention

        scores = torch.matmul(Q, K.permute(0,2,1)) / self.scale

        print("shape of the score",scores.shape)

        attention = F.softmax(scores, dim=-1)
        #print("shape of atte",attention.shape)

        out = torch.matmul(attention, V)
        #print("shape of out",out.shape)

        

        return out
    
selfA = SelfAttention(64)

x = torch.randn(1,1, 64)

self_out = selfA(x)
print('shape of selfAtt',self_out.shape)  # should output torch.Size([1, 64,1])

class cross_mha_add(nn.Module):
    # default dim of the model is 8 and head is 4
    def __init__(self, d_model=8, num_heads=4, dropout=0.01):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads

        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        ##create a list of layers for K, and a list of layers for V
       
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])

        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):

        Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        scores = Q_K_matmul/m.sqrt(self.d)

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, x,x2):

        Q = [linear_Q(x2) for linear_Q in self.linear_Qs]
        K = [linear_K(x) for linear_K in self.linear_Ks]
        V = [linear_V(x) for linear_V in self.linear_Vs]
        #print('shape of Value',V[0].shape)


        output_per_head = []
        attn_weights_per_head = []

        for Q_, K_, V_ in zip(Q, K, V):
           
            ##run scaled_dot_product_attention
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_)

            # shape(output) = [B x feature_dim x D/num_heads]
            # shape(attn_weights_per_head) = [B x feature_dim x feature_dim]
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)
        print('shape of attnention weights',attn_weight[0].shape)


        output = torch.cat(output_per_head, -1)

        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)

        projection = self.dropout(self.mha_linear(output))

        return projection#, attn_weights

# Define input feature vectors
x1 = torch.randn(1, 1,64)
x2 = torch.randn(1, 1,64)

# Define cross-attention layer
cross_attn = cross_mha_add(64,64)
# Apply cross-attention to input feature vectors( those are flttened vectors came form mlp and cnn )
out = cross_attn(self_out, self_out)
# Check shape of output tensor
print('shape of method 1',out.shape)  # should output torch.Size([1, 64,1])
