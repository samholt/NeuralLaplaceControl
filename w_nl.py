import logging
import os
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchlaplace import laplace_reconstruct
import time
import logging
from config import get_config, CME_reconstruction_terms
from copy import deepcopy
from overlay import setup_logger, create_env, generate_irregular_data_delay_time_multi, get_val_loss_delay_time_multi, get_val_loss_delay_precomputed, compute_val_data_delay
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()


class ReverseGRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self, dimension_in, latent_dim, hidden_units, encode_obs_time=True):
        super(ReverseGRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.gru = nn.GRU(dimension_in, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latent_dim)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data):
        trajs_to_encode = observed_data  # (batch_size, t_observed_dim, observed_dim)
        reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1,))
        out, _ = self.gru(reversed_trajs_to_encode)
        return self.linear_out(out[:, -1, :])


class LaplaceRepresentationFunc(nn.Module):
    # SphereSurfaceModel : C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords :
    # b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(LaplaceRepresentationFunc, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        phi_max = torch.pi / 2.0
        self.phi_scale = phi_max - -torch.pi / 2.0

    def forward(self, i):
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        theta = nn.Tanh()(out[:, : self.output_dim, :]) * torch.pi  # From - pi to + pi
        phi = (
            nn.Tanh()(out[:, self.output_dim :, :]) * self.phi_scale / 2.0 - torch.pi / 2.0 + self.phi_scale / 2.0
        )  # Form -pi / 2 to + pi / 2
        return theta, phi


class NeuralLaplaceModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        hidden_units=64,
        s_recon_terms=33,
        ilt_algorithm="fourier",
        encode_obs_time=False,
        state_mean=None,
        state_std=None,
        action_mean=None,
        action_std=None,
        normalize=False,
        normalize_time=False,
        dt=0.05,
    ):
        super(NeuralLaplaceModel, self).__init__()
        self.ilt_algorithm = ilt_algorithm
        if ilt_algorithm == "cme":
            terms = CME_reconstruction_terms()
            s_recon_terms = terms[np.argmin(terms < s_recon_terms) - 2]
        action_encoder_latent_dim = 2
        laplace_latent_dim = state_dim + action_encoder_latent_dim
        self.latent_dim = latent_dim
        self.action_encoder = ReverseGRUEncoder(
                action_dim,
                action_encoder_latent_dim,
                hidden_units // 2,
                encode_obs_time=encode_obs_time,
            )
        self.laplace_rep_func = LaplaceRepresentationFunc(
            s_recon_terms, state_dim, laplace_latent_dim, hidden_units=hidden_units
        )
        self.encode_obs_time = encode_obs_time
        self.output_dim = state_dim
        self.normalize = normalize
        self.normalize_time = normalize_time
        self.s_recon_terms = s_recon_terms
        # if self.encode_obs_time:
        #     state_mean = np.concatenate((state_mean,np.array([1])))
        #     state_std = np.concatenate((state_std,np.array([1])))
        #     action_mean = np.concatenate((action_mean,np.array([1])))
        #     action_std = np.concatenate((action_std,np.array([1])))
        self.register_buffer("state_mean", torch.tensor(state_mean))
        self.register_buffer("state_std", torch.tensor(state_std))
        self.register_buffer("action_mean", torch.tensor(action_mean))
        self.register_buffer("action_std", torch.tensor(action_std))
        self.register_buffer("dt", torch.tensor(dt))

    def forward(self, in_batch_obs, in_batch_action, ts_pred):
        # in_batch_action = in_batch_action[:,:1,:]
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
            if self.normalize_time:
                ts_pred = ts_pred / (self.dt * 8.0)  # pyright: ignore
                # ts_pred = (((ts_pred - self.dt) / (self.dt*4.0)) + 0.05)
            # batch_action = in_batch_action.view(-1, in_batch_action.shape[2])
            # batch_action = batch_action.view(*in_batch_action.shape)
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        # p_action = batch_action.view(batch_action.shape[0],batch_action.shape[-1])
        if len(batch_action.shape) == 2:
            batch_action = batch_action.unsqueeze(1)
        p_action = self.action_encoder(batch_action)
        sa_in = torch.cat((batch_obs, p_action), axis=1)
        p = sa_in
        return torch.squeeze(
            laplace_reconstruct(
                self.laplace_rep_func,
                p,
                ts_pred,
                recon_dim=self.output_dim,
                ilt_algorithm=self.ilt_algorithm,
                ilt_reconstruction_terms=self.s_recon_terms,
            )
        )


def load_replay_buffer(fn):
    offline_dataset = np.load(fn, allow_pickle=True).item()
    return offline_dataset
