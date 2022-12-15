# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:30:56 2022

@author: Omnia
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import math as m
# %%
class MultiHeadAttention(nn.Module):
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
        # shape(Q) = [B x feature_dim x D/num_heads] = [B x T x d_k]
        # shape(K, V) = [B x T x d_k]

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

            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)
        print('shape of attnention weights',attn_weight[0].shape)

        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        
        projection = self.dropout(self.mha_linear(output))

        return projection#, attn_weights

# %%

model_urls = {
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}



class PathNet(nn.Module):

    def __init__(self, features ,path_dim=32, act=None, num_classes=3):
        super(PathNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.attention = MultiHeadAttention(d_model=49, num_heads=7)
        self.classifier = nn.Sequential(nn.Linear(512 * 49, 1024))
        self.linear = nn.Linear(1024, 3)
        #self.act = act



    def forward(self,x):
        
        x = self.features(x)
        print('shape of features',x.shape)
        x = self.avgpool(x)
        print('shape of avgpool',x.shape)
        
        x = x.view(x.size(0), -1,x.size(2) *x.size(3))
        print('shape oftr mul',x.shape)
        x = self.attention(x)
        print('shape of atten',x.shape)
        x = x.view(x.size(0), -1)
        print('shape oftr flat',x.shape)


        x = self.classifier(x)
        #features = self.classifier(x)

        print('shape of fea',x.shape)
        hazard = self.linear(x)

        #return  features,hazard
        return  hazard
        #return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True, label_dim=3, pretrained=True, progress=True):
    model = PathNet(make_layers(cfgs[cfg], batch_norm=batch_norm), act=act, num_classes=label_dim)
    
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        for key in list(pretrained_dict.keys()):
            if 'classifier' in key: pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Initializing Path Weights")

    return model
#%%
#model = get_vgg()

# from torchsummary import summary 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model.to(device=DEVICE,dtype=torch.float)
# summary(model,(3, 224, 224))
