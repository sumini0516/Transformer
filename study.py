import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size nees to div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # how many example we send in at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = keys.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        # qureies shape : (N, qurey_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy : (N, heads, qurey_len, key_len) #query -> target, key -> source sentence
        #                                                -> each word in our target how much pay attention to each word in input

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape : (N, heads, qurey_len, key_len)
        # values shape : (N, value_len, heads, heads_dim) -> key_len==value_len (always)
        # (N,qurey_len, heads, head_dim) then flatten last two dimension

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
