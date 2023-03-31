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
from attrdict import AttrDict

from models.modules import PoolingEncoder, Decoder

class CNP(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        self.enc1 = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.enc2 = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=2*dim_hid,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, num_samples=None):
        encoded = torch.cat([self.enc1(xc, yc), self.enc2(xc, yc)], -1)
        encoded = torch.stack([encoded]*xt.shape[-2], -2)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()
        py = self.predict(batch.xc, batch.yc, batch.x)
        ll = py.log_prob(batch.y).sum(-1)

        if self.training:
            outs.loss = -ll.mean()
        else:
            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                outs.ctx_ll = ll[...,:num_ctx].mean()
                outs.tar_ll = ll[...,num_ctx:].mean()
            else:
                outs.ctx_ll = ll[...,:num_ctx]
                outs.tar_ll = ll[...,num_ctx:]

        return outs
