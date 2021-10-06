import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .layers import PatchEmbedding, TransformerEncoder, ClassificationHead

class ViT(nn.Sequential):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224, depth = 12, n_classes = 10, **kwargs):
        super().__init__(
            PatchEmbedding( in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size = emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
        
        
