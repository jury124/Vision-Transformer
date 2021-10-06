# einops

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# Pytorch

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = patch_size, stride = patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)

        x = torch.cat([cls_tokens, x],dim=1)

        x += self.positions

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 768, num_head = 8, dropout=0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.keys = nn.Linear(emb_size,emb_size)
        self.queries = nn.Linear(emb_size,emb_size)
        self.values = nn.Linear(emb_size,emb_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size,emb_size)
    
    def forward(self, x, mask=None):

        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h = self.num_head) # n = query length
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h = self.num_head) # n = key length
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h = self.num_head)
        
        energy = torch.einsum('bhqd,bhkd -> bhqk', queries, keys) #

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        attn = F.softmax(energy, dim=-1) / scaling
        attn = self.dropout(attn)

        out = torch.einsum('bhal, bhlv -> bhav',attn , values)
        out = rearrange(out, 'b h n d -> b n (h d)')


        out = self.projection(out)
        

        
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, dropout = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size = 768, dropout = 0., forward_expansion = 4, forward_dropout=0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion = forward_expansion, dropout = forward_dropout),
                nn.Dropout(dropout)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction = 'mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size,n_classes)
        )