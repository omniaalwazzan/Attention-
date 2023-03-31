import torch
import torch.nn.functional as F
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CrossAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = torch.nn.Linear(d_model, d_model, bias=False)
        self.key = torch.nn.Linear(d_model, d_model, bias=False)
        self.value = torch.nn.Linear(d_model, d_model, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, x1, x2):
        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)

        # Calculate scaled dot product attention
        scores = torch.bmm(Q, K.permute(0,2, 1)) / self.scale
        print('Attention score shape',scores.shape)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)

        return out


        ### METHOD 1 ###
# Define input feature vectors
x1 = torch.randn(1, 64,1)
x2 = torch.randn(1, 64,1)

# Define cross-attention layer
cross_attn = CrossAttention(1)
# Apply cross-attention to input feature vectors( those are flttened vectors came form mlp and cnn )
out = cross_attn(x1, x2)
# Check shape of output tensor
print('shape of method 1',out.shape)  # should output torch.Size([1, 64,1])


        ### METHOD 2 ###
# Define input feature vectors
x3 = torch.randn(1, 1,64)
x4 = torch.randn(1, 1,64)
# Define cross-attention layer, we send the dim of the vector
cross_attn_ = CrossAttention(64)
# Apply cross-attention to input feature vectors( those are flttened vectors came form mlp and cnn )
out_ = cross_attn_(x3, x4)
#Check shape of output tensor
print('shape of method 2',out_.shape)  # should output torch.Size([1, 1,64])
