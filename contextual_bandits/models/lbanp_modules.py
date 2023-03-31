# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# Copyright (c) 2021, Phil Wang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen 
# and the Perceiver (https://arxiv.org/abs/2103.03206) implementation
# from https://github.com/lucidrains/Perceiver-pytorch by Phil Wang
####################################################################################

import torch
import torch.nn as nn

from torch import nn, einsum
from torch.nn import ModuleList
import copy
from einops import rearrange, repeat
import torch.nn.functional as F


def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs) + x

class PostNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = x + self.fn(x, **kwargs)

        return self.norm(x)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_feedforward=128, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_feedforward * 2),
            GEGLU(),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)





class LBANPEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(LBANPEncoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)


        if norm_first:
            self.latent_self_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.latent_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.d_model)
            self.cross_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.latent_self_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.latent_ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.d_model)
            self.cross_ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))

    def forward(self, context_encodings, latents):
        x = latents
        x = self.cross_attn(x, context = context_encodings)
        x = self.cross_ff(x)

        x = self.latent_self_attn(x)
        x = self.latent_ff(x)

        return x



class LBANPEncoder(nn.Module):
    """
        Iterative Attention-based model that encodes context datapoints into a list of embeddings
    """
    def __init__(self, encoder_layer, num_layers):
        super(LBANPEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, context_encodings, latents):
        b, *axis = context_encodings.shape
        latents = repeat(latents, 'n d -> b n d', b = b)

        layer_outputs = []
        for layer in self.layers:
            latents = layer(context_encodings, latents)
            layer_outputs.append(latents)
        return layer_outputs



class NPDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(NPDecoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)

        if norm_first:
            self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.latent_dim)
            self.cross_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.cross_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.latent_dim)
            self.cross_ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))


    def forward(self, query_encodings, context):
        
        x = query_encodings
        x = self.cross_attn(x, context = context)
        x = self.cross_ff(x)

        return x

class NPDecoder(nn.Module):
    """
        Attention-based model that retrieves information via the context encodings to make predictions for the query/target datapoints
    """
    def __init__(self, decoder_layer, num_layers):
        super(NPDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query_encodings, context_encodings):
        assert len(context_encodings) == self.num_layers

        x = query_encodings
        for layer, context_enc in zip(self.layers, context_encodings):
            x = layer(x, context=context_enc)

        out = x
        return out




class TransformerEncoder(nn.Module):
    """
        Transformer-based model that encodes context datapoints into a list of embeddings
    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        layer_outputs = [output]
        for layer in self.layers:
            output = layer(output)
            layer_outputs.append(output)
        return layer_outputs




class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(TransformerEncoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)

        if norm_first:
            self.dataset_self_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.dataset_self_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))

    def forward(self, context):
        x = context
        x = self.dataset_self_attn(x)
        x = self.ff(x)
        return x