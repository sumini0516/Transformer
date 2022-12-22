import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, LayerNorm
import math
import numpy as np


class WordEncoding(nn.Module):
    def __init__(self, embed_weights, d_model):
        # super(클래스, self).__init__(): 자식 클래스가 상속받는 부모 클래스를 자식 클래스에 불러오겠다는 의미
        super(WordEncoding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_weights, freeze=False, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x)


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, device=None):
        super(PositionEncoding, self).__init__()
        self.device = device
        # torch.zeros(a,b): 0값을 가지는 a x b tensor를 생성
        self.positions_emb = torch.zeros(max_len, d_model)
        # torch.arange: array생성
        # unsqueeze(1): 지정한 dimension 자리에 size가 1인 빈 공간을 채워주면서 차원을 확장해준다
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(1e4) / d_model))
        self.positions_emb[:, 0::2] = torch.sin(position * div_term)
        self.positions_emb[:, 1::2] = torch.cos(position * div_term)

    def forward(self, inputs):
        outputs = self.positions_emb[:inputs.size(1), :]
        outputs = outputs.unsqueeze(0).repeat(inputs.size(0), 1, 1).to(self.device)
        return outputs


