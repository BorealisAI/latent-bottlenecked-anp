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

def gather(items, idxs):
    K = idxs.shape[0]
    idxs = idxs.to(items[0].device)
    gathered = []
    for item in items:
        gathered.append(torch.gather(
            torch.stack([item]*K), -2,
            torch.stack([idxs]*item.shape[-1], -1)).squeeze(0))
    return gathered[0] if len(gathered) == 1 else gathered

def sample_subset(*items, r_N=None, num_samples=None):
    r_N = r_N or torch.rand(1).item()
    K = num_samples or 1
    N = items[0].shape[-2]
    Ns = min(max(1, int(r_N * N)), N-1)
    batch_shape = items[0].shape[:-2]
    idxs = torch.rand((K,)+batch_shape+(N,)).argsort(-1)
    return gather(items, idxs[...,:Ns]), gather(items, idxs[...,Ns:])

def sample_with_replacement(*items, num_samples=None, r_N=1.0, N_s=None):
    K = num_samples or 1
    N = items[0].shape[-2]
    N_s = N_s or max(1, int(r_N * N))
    batch_shape = items[0].shape[:-2]
    idxs = torch.randint(N, size=(K,)+batch_shape+(N_s,))
    return gather(items, idxs)

def sample_mask(B, N, num_samples=None, min_num=3, prob=0.5):
    min_num = min(min_num, N)
    K = num_samples or 1
    fixed = torch.ones(K, B, min_num)
    if N - min_num > 0:
        rand = torch.bernoulli(prob*torch.ones(K, B, N-min_num))
        mask = torch.cat([fixed, rand], -1)
        return mask.squeeze(0)
    else:
        return fixed.squeeze(0)