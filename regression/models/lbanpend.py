# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen 
####################################################################################


import torch
import torch.nn as nn

from models.modules import build_mlp
from attrdict import AttrDict

from torch.distributions.normal import Normal

from models.lbanp_modules import LBANPEncoderLayer, LBANPEncoder, LBANPDecoderLayer, LBANPDecoder
from models.lbanp import LBANP


class LBANPEND(LBANP):
    """
        Latent Bottlenecked Attentive Neural Processes (Efficient Not-Diagonal variant)
    """
    def __init__(
        self,
        num_latents,  
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        num_std_layers,
        norm_first=True,
        bound_std=False,
        cov_approx='cholesky',
        prj_dim=5,
        prj_depth=4,
        diag_depth=4
    ):
        super(LBANPEND, self).__init__(
            num_latents,
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            norm_first,
            bound_std
        )

        assert cov_approx in ['cholesky', 'lowrank']
        self.cov_approx = cov_approx
        
        # Mean Network

        self.mean_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y)
        )

        # Std Network

        # Learnable latents for iterative attention
        self.std_latents = nn.Parameter(torch.randn(num_latents, self.latent_dim), requires_grad=True) 
        
        std_encoder_layer = LBANPEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=norm_first, batch_first=True)
        self.std_encoder = LBANPEncoder(std_encoder_layer, num_std_layers)

        std_decoder_layer = LBANPDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=norm_first, batch_first=True)
        self.std_decoder = LBANPDecoder(std_decoder_layer, num_std_layers)

        self.projector = build_mlp(d_model, dim_feedforward, prj_dim*dim_y, prj_depth)

        if cov_approx == 'lowrank':
            self.diag_net = build_mlp(d_model, dim_feedforward, dim_y, diag_depth)

    def decode(self, out_encoder, batch_size, dim_y, num_target):
        mean = self.mean_net(out_encoder).view(batch_size, -1)


        out_std_context_encodings = self.std_encoder(out_encoder, self.std_latents)
        
        out_std_decoder = self.std_decoder(out_encoder, out_std_context_encodings)


        std_prj = self.projector(out_std_decoder)
        std_prj = std_prj.view((batch_size, num_target*dim_y, -1))
        
        if self.cov_approx == 'cholesky':
            std_tril = torch.bmm(std_prj, std_prj.transpose(1,2))
            std_tril = std_tril.tril()
            if self.bound_std:
                diag_ids = torch.arange(num_target*dim_y, device='cuda')
                std_tril[:, diag_ids, diag_ids] = 0.05 + 0.95*torch.tanh(std_tril[:, diag_ids, diag_ids])
            pred_tar = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale_tril=std_tril)
        else:
            diagonal = torch.exp(self.diag_net(out_encoder)).view((batch_size, -1, 1))
            std = torch.bmm(std_prj, std_prj.transpose(1,2)) + torch.diag_embed(diagonal.squeeze(-1))
            pred_tar = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=std)

        return pred_tar

    def forward(self, batch, reduce_ll=True):
        batch_size = batch.x.shape[0]
        dim_y = batch.y.shape[-1]
        num_target = batch.xt.shape[1]

        out_encoder = self.get_predict_encoding(batch)
        
        pred_tar = self.decode(out_encoder, batch_size, dim_y, num_target)

        outs = AttrDict()
        outs.tar_ll = pred_tar.log_prob(batch.yt.reshape(batch_size, -1))

        if not self.training:
            outs.tar_ll /= num_target

        if reduce_ll:
            outs.tar_ll = outs.tar_ll.mean()
            outs.loss = - (outs.tar_ll)
            outs.mean_std = torch.mean(pred_tar.covariance_matrix)
        else:
            outs.tar_ll = outs.tar_ll.unsqueeze(-1)

        return outs


    def predict(self, xc, yc, xt, num_samples=50, return_samples=False):
        batch_size = xc.shape[0]
        dim_y = yc.shape[-1]
        num_target = xt.shape[1]

        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device='cuda')

        out_encoder = self.get_predict_encoding(batch)
        pred_tar = self.decode(out_encoder, batch_size, dim_y, num_target)
            
        yt_samples  = pred_tar.rsample([num_samples]).view(num_samples, batch_size, num_target, -1)
        if return_samples:
            return yt_samples

        std = yt_samples.std(dim=0)
        return Normal(pred_tar.mean.view(batch_size, num_target, -1), std)


    def sample(self, xc, yc, xt, num_samples=50):
        return self.predict(xc, yc, xt, num_samples, return_samples=True)