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

from models.canp import CANP
from utils.misc import stack, logmeanexp
from utils.sampling import sample_with_replacement as SWR, sample_subset


class BANP(CANP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec.add_ctx(2 * kwargs['dim_hid'])

    def encode(self, xc, yc, xt, mask=None):
        theta1 = self.enc1(xc, yc, xt)
        theta2 = self.enc2(xc, yc)
        encoded = torch.cat([theta1,
                             torch.stack([theta2] * xt.shape[-2], -2)], -1)
        return encoded

    def predict(self, xc, yc, xt, num_samples=None, return_base=False):
        # botorch 사용하기 위해 추가된 statement
        if xc.shape[-3] != xt.shape[-3]:
            xt = xt.transpose(-3, -2)

        with torch.no_grad():
            bxc, byc = SWR(xc, yc, num_samples=num_samples)
            sxc, syc = stack(xc, num_samples), stack(yc, num_samples)

            encoded = self.encode(bxc, byc, sxc)
            py_res = self.dec(encoded, sxc)

            mu, sigma = py_res.mean, py_res.scale
            res = SWR((syc - mu) / sigma).detach()
            res = (res - res.mean(-2, keepdim=True))

            bxc = sxc
            byc = mu + sigma * res

            del sxc, mu, sigma, res

        encoded_base = self.encode(xc, yc, xt)
        del xc, yc

        sxt = stack(xt, num_samples)
        encoded_bs = self.encode(bxc, byc, sxt)
        del bxc, byc

        py = self.dec(stack(encoded_base, num_samples),
                      sxt, ctx=encoded_bs)
        del sxt, encoded_bs

        if self.training or return_base:
            py_base = self.dec(encoded_base, xt)
            return py_base, py
        else:
            return py

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()

        def compute_ll(py, y):
            ll = py.log_prob(y).sum(-1)
            if ll.dim() == 3 and reduce_ll:
                ll = logmeanexp(ll)
            return ll

        if self.training:
            py_base, py = self.predict(batch.xc, batch.yc, batch.x,
                                       num_samples=num_samples)

            outs.ll_base = compute_ll(py_base, batch.y).mean()
            outs.ll = compute_ll(py, batch.y).mean()
            outs.loss = - outs.ll_base - outs.ll
        else:
            py = self.predict(batch.xc, batch.yc, batch.x,
                              num_samples=num_samples)
            ll = compute_ll(py, batch.y)
            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                outs.ctx_loss = ll[..., :num_ctx].mean()
                outs.tar_loss = ll[..., num_ctx:].mean()
            else:
                outs.ctx_loss = ll[..., :num_ctx]
                outs.tar_loss = ll[..., num_ctx:]

        return outs
