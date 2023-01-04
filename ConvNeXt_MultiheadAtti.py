# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:32:46 2022

@author: omnia
"""
import timm
import torch.nn as nn
import math as m
import torch
import torch.nn.functional as F
from torchsummary import summary


class MultiHeadAttention(nn.Module):
    # default values for the diminssion of the model is 8 and heads 4
    def __init__(self, d_model=8, num_heads=4, dropout=0.1):
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
        # shape(Q) = [B x feature_dim x D/num_heads] = [B x feature_dim x d_k]
        # shape(K, V) = [B x feature_dim x d_k]

        #Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        Q_K_matmul = Q.view(Q.shape[0], Q.shape[1], -1)  * K.view(K.shape[0], K.shape[2],K.shape[1])
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
        #print('shape of Query',Q[0].shape)
        K = [linear_K(x) for linear_K in self.linear_Ks]
        #print('shape of Key',K[0].shape)        
        V = [linear_V(x) for linear_V in self.linear_Vs]
        #print('shape of Value',V[0].shape)

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
        #print('shape of attnention weights',attn_weight[0].shape)

        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        # shape(output) = [B x feature_dim x D]
        # shape(attn_weights) = [B x num_heads x feature_dim x feature_dim]
        
        projection = self.dropout(self.mha_linear(output))

        return projection#, attn_weights


def load_model():

    model =  timm.create_model('convnext_base', pretrained=True,num_classes=32) 


    # Disable gradients on all model parameters to freeze the weights
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = False

    # Unfreeze the last stage
    for param in model.stages[3].parameters():
        param.requires_grad = True
    
    return model

cnn_m = load_model()
# x3 = torch.randn(2,3,224,224)
# out = cnn_m(x3)

# out_s = torch.unsqueeze(out, 2)

# y_projection = MultiHeadAttention(d_model=1, num_heads=1)

# ex1 = y_projection(out_s)


class MyEnsemble(nn.Module):
    def __init__(self, nb_classes=3):
        super(MyEnsemble, self).__init__()
        self.model_image =  cnn_m
        self.attention = MultiHeadAttention(d_model=1, num_heads=1)
        self.fc = nn.Linear(32*1, 32)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Linear(32, nb_classes)
        
        #self.conv_stack= conv_(4,1)
    
    def forward(self, x):

        x = self.model_image(x)
        x = torch.unsqueeze(x, 2)
        print(x.shape)
        x = self.attention(x)
        print('shape of atten',x.shape)
        x = x.view(x.size(0), -1)
        print('shape oftr flat',x.shape)

        #x1 = x1.view(x1.size(0), -1)
               
   
        #print('shape after conv', x.shape)
        #x = x.flatten(start_dim=1)
        
        #print('shape aftr flatten', x.shape)

        #x = torch.add(x_add,x_pro)
        
        x = self.fc(x)
        #print('fc after combined', x.shape)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
    
    
def MOAB() -> MyEnsemble:
    model = MyEnsemble()
    return model


# from torchsummary import summary 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MOAB().to(device=DEVICE,dtype=torch.float)
summary(MOAB(),(3, 224, 224))
