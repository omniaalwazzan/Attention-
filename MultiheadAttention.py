# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:46:04 2022

@author: omnia
"""


# %%
import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

# %%
class MultiHeadAttention(nn.Module):
    # default dim of the model is 8 and head is 4
    def __init__(self, d_model=8, num_heads=4, dropout=0.3):
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
        # D is the dim of the model ,d_k = D/num_heads, B is the batch size
        # shape(Q) = [B x feature_dim x D/num_heads] = [B x feature_dim x d_k]
        # shape(K, V) = [B x feature_dim x d_k]

        Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        scores = Q_K_matmul/m.sqrt(self.d)
        # shape(scores) = [B x feature_dim x feature_dim]

        attention_weights = F.softmax(scores, dim=-1)
        # shape(attention_weights) = [B x feature_dim x feature_dim]

        output = torch.matmul(attention_weights, V)
        # shape(output) = [B x feature_dim x D/num_heads]

        return output, attention_weights

    def forward(self, x):
        # shape(x) = [B x feature_dim x D]

        Q = [linear_Q(x) for linear_Q in self.linear_Qs]
        print('shape of Query',Q[0].shape)
        K = [linear_K(x) for linear_K in self.linear_Ks]
        print('shape of Key',K[0].shape)        
        V = [linear_V(x) for linear_V in self.linear_Vs]
        print('shape of Value',V[0].shape)

        # shape(Q, K, V) = [B x feature_dim x D/num_heads] * num_heads

        output_per_head = []
        attn_weights_per_head = []
        # shape(output_per_head) = [B x feature_dim x D/num_heads] * num_heads
        # shape(attn_weights_per_head) = [B x feature_dim x feature_dim] * num_heads
        for Q_, K_, V_ in zip(Q, K, V):
           
            ##run scaled_dot_product_attention
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_)

            # shape(output) = [B x feature_dim x D/num_heads]
            # shape(attn_weights_per_head) = [B x feature_dim x feature_dim]
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)
        print('shape of attnention weights',attn_weight[0].shape)
        print('shape of out weights',output[0].shape)


        output = torch.cat(output_per_head, -1)
        print('shape of out after cat',output[0].shape)

        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        # shape(output) = [B x feature_dim x D]
        # shape(attn_weights) = [B x num_heads x feature_dim x feature_dim]
       
        projection = self.dropout(self.mha_linear(output))

        return projection, attn_weights
# %%
x = torch.randn(1,1,64)
y = torch.randn(3,512,49)
z = torch.randn(3,197,768)
projection = MultiHeadAttention(d_model=768, num_heads=12)
y_projection = MultiHeadAttention(d_model=49, num_heads=7)

ex1 = y_projection(y)


# x = torch.randn(1, 3, 224, 224)
# # 2D conv
# conv = nn.Conv2d(3, 768, 16, 16) # in_channel =3, out_ch = (3x16x16)=768, k= 16, s =16
# conv(x).reshape(-1, 196).transpose(0,1).shape
