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
from torch.distributions.normal import Normal
from attrdict import AttrDict

from models.tnp import TNP


class TNPA(TNP):
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
        drop_y=0.5,
        pretrain=False
    ):
        super(TNPA, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            drop_y,
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

        self.pretrain = pretrain

    def forward(self, batch, reduce_ll=True):
        if self.training and self.pretrain:
            return self.forward_pretrain(batch)
        num_ctx, num_all = batch.xc.shape[1], batch.x.shape[1]

        out_encoder = self.encode(batch, autoreg=True, drop_ctx=True)
        out_encoder = torch.cat((out_encoder[:, :num_ctx], out_encoder[:, num_all:]), dim=1)
        out = self.predictor(out_encoder)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(std)

        pred_dist = Normal(mean, std)
        loss = - pred_dist.log_prob(batch.y).sum(-1).mean()
        
        outs = AttrDict()
        outs.loss = loss
        return outs

    def forward_pretrain(self, batch):
        num_points = batch.x.shape[1]
        out_encoder = self.encode(batch, autoreg=True, drop_ctx=True, pretrain=True)
        out_encoder = out_encoder[:, num_points:]
        out = self.predictor(out_encoder)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(std)

        pred_dist = Normal(mean, std)
        loss = - pred_dist.log_prob(batch.y[:, 1:]).sum(-1).mean()
        
        outs = AttrDict()
        outs.loss = loss
        return outs 

    def predict(self, xc, yc, xt):
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device='cuda')

        num_context = xc.shape[1]

        # in evaluation tnpa = tnpd because we only have 1 target point to predict
        out_encoder = self.encode(batch, autoreg=False, drop_ctx=False)
        out = self.predictor(out_encoder)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(std)
        mean, std = mean[:, num_context:, :], std[:, num_context:, :]

        outs = AttrDict()
        outs.loc = mean.unsqueeze(0)
        outs.scale = std.unsqueeze(0)
        outs.ys = Normal(outs.loc, outs.scale)
        
        return outs