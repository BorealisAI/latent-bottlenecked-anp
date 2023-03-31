# MIT License

# Copyright (c) 2022 Tung Nguyen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn

from models.modules import build_mlp


class TNP(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        drop_y=0.5
    ):
        super(TNP, self).__init__()

        self.drop_y = drop_y
        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def drop(self, y):
        y_dropped = torch.randn_like(y)
        not_drop_ids = torch.rand_like(y) > self.drop_y
        y_dropped[not_drop_ids] = y[not_drop_ids]
        return y_dropped

    def construct_input(self, batch, autoreg=False, drop_ctx=False):
        if drop_ctx:
            yc_dropped = self.drop(batch.yc)
            x_y_ctx = torch.cat((batch.xc, yc_dropped), dim=-1)
        else:
            x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        x_0_tar = torch.cat((batch.xt, torch.zeros_like(batch.yt)), dim=-1)
        if not autoreg:
            inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        else:
            x_y_tar = torch.cat((batch.xt, batch.yt), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1)
        return inp

    def create_mask(self, batch, autoreg=False):
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar
        if not autoreg:
            mask = torch.zeros(num_all, num_all, device='cuda')
            mask[:, num_ctx:] = float('-inf')
        else:
            mask = torch.zeros((num_all+num_tar, num_all+num_tar), device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0 # all points attend to context points
            mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1) # each real target point attends to itself and precedding real target points
            mask[num_all:, num_ctx:num_all].triu_(diagonal=0) # each fake target point attends to preceeding real target points

        return mask

    def construct_input_pretrain(self, batch, drop_y):
        if drop_y:
            y = self.drop(batch.y)
        else:
            y = batch.y
        x_y = torch.cat((batch.x, y), dim=-1)
        x_0 = torch.cat((batch.x, torch.zeros_like(batch.y)), dim=-1)[:, 1:]
        inp = torch.cat((x_y, x_0), dim=1)
        return inp

    def create_mask_pretrain(self, batch):
        num_points = batch.x.shape[1]

        mask = torch.zeros((2*num_points-1, 2*num_points-1), device='cuda').fill_(float('-inf'))
        mask[:num_points, :num_points].triu_(diagonal=1)
        mask[num_points:, 1:num_points].triu_(diagonal=0)
        mask[num_points:, 0] = 0.0

        return mask

    def encode(self, batch, autoreg=False, drop_ctx=False, pretrain=False):
        if not pretrain:
            inp = self.construct_input(batch, autoreg, drop_ctx)
            mask = self.create_mask(batch, autoreg)
        else:
            inp = self.construct_input_pretrain(batch, drop_ctx)
            mask = self.create_mask_pretrain(batch)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask)
        return out