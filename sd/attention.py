import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module): 
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True): 
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # Wk, Wq, Wv integrated in one Linear layer. 
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias) # W_out, from softmax output to final output
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # dimension of each head
    
    def forward(self, x: torch.Tensor, causal_mask=False): 
        # x: (Batch_Size, Seq_Len, Dim)

        input_shape = x.shape

        batch_size, sequence_legth, d_embed = input_shape

        interm_shape = (batch_size, sequence_legth, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> project: (Batch_Size, Seq_Len, Dim * 3) -> chunk: 3 tensors of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask: 
            # Mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output

class CrossAttention(nn.Module): 

    def __init__(self, n_heads: int, d_embd: int, d_cross: int, in_proj_bias=True, out_proj_bias=True): 
        super().__init__()
        self.q_proj = nn.Linear(d_embd, d_embd, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=in_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads
    
    def forward(self, x, y): 
        # x: (latent): (Batch_Size, Seq_Len_Q, Dim_Q)
        # y: (context): (Batch_Size, Seq_Len_K or V, Dim_K or V) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embd = input_shape

        interm_shape = (batch_size, -1, self.n_heads, self.d_head) # -1: auto adapt

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(d_embd)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output



