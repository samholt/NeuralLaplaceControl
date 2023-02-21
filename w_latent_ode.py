import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torchlaplace import laplace_reconstruct
import time
import logging
from config import get_config
from copy import deepcopy
from overlay import setup_logger, create_env, generate_irregular_data_delay, get_val_loss_delay, get_val_loss_delay_precomputed, compute_val_data_delay
# from oracle import cartpole_dynamics
from tqdm import tqdm
from experiments.baseline_models.original_latent_ode import GeneralLatentODEOfficial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()

from copy import deepcopy
from time import strftime, time

import keyboard
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch import Tensor, nn
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor
from tqdm import tqdm

matplotlib.use("Agg")
import argparse
import datetime
import time
from random import SystemRandom

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import model_selection
from torch.nn.functional import relu

from experiments.baseline_models.latent_ode_lib.create_latent_ode_model import create_LatentODE_model_direct
from experiments.baseline_models.latent_ode_lib.diffeq_solver import DiffeqSolver
from experiments.baseline_models.latent_ode_lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from experiments.baseline_models.latent_ode_lib.ode_rnn import *
from experiments.baseline_models.latent_ode_lib.parse_datasets import parse_datasets
from experiments.baseline_models.latent_ode_lib.plotting import *
from experiments.baseline_models.latent_ode_lib.rnn_baselines import *
from experiments.baseline_models.latent_ode_lib.utils import compute_loss_all_batches_direct

class GeneralLatentODEOfficial(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        hidden_units=64,
        state_mean=None,
        state_std=None,
        action_mean=None,
        action_std=None,
        normalize=False,
        normalize_time=False,
        dt=0.05,
        classif_per_tp=False,
        n_labels=1,
        obsrv_std=0.01,
    ):
        super(GeneralLatentODEOfficial, self).__init__()
        input_dim = state_dim + action_dim
        action_encoder_latent_dim = 2
        latents = state_dim + action_encoder_latent_dim
        # latents = 2
        self.latents = latents
        self.output_dim = state_dim
        self.normalize = normalize
        self.normalize_time = normalize_time
        self.register_buffer("state_mean", torch.tensor(state_mean))
        self.register_buffer("state_std", torch.tensor(state_std))
        self.register_buffer("action_mean", torch.tensor(action_mean))
        self.register_buffer("action_std", torch.tensor(action_std))
        self.register_buffer("dt", torch.tensor(dt))

        obsrv_std = torch.Tensor([obsrv_std]).to(device)
        z0_prior = Normal(
            torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device)
        )
        self.model = create_LatentODE_model_direct(
            input_dim,
            z0_prior,
            obsrv_std,
            device,
            classif_per_tp=classif_per_tp,
            n_labels=n_labels,
            latents=latents,
            units=hidden_units,
            gru_units=hidden_units,
        ).to(device)
        self.latents = latents
        self.batch_obs_buffer = torch.zeros(1,4,state_dim).to(device)

    def _get_loss(self, dl):
        loss = compute_loss_all_batches_direct(self.model, dl, device=device, classif=0)
        return loss["loss"], loss["mse"]

    def train_loss(self, in_batch_obs, in_batch_action, ts_pred):
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        p_action = self.action_encoder(batch_action)
        sa_in = torch.cat((batch_obs, p_action), axis=1)
        if len(sa_in.shape) == 2:
            sa_in = sa_in.unsqueeze(1)

        # p = self.obs_encoder(sa_in)
        p = sa_in.squeeze()
        return torch.squeeze(laplace_reconstruct(
            self.laplace_rep_func, p, ts_pred, recon_dim=self.output_dim, ilt_algorithm=self.ilt_algorithm,
        ))

    def train_step(self, in_batch_obs, in_batch_action, ts_pred, observed_tp, pred_batch_obs_diff):
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        # if self.normalize_time:
        #     ts_pred = (ts_pred / (self.dt*8.0))
        batch_size = batch_obs.shape[0]
    
        if len(batch_action.shape) == 2:
            batch_action = batch_action.unsqueeze(1)

        observed_data = torch.cat((batch_obs, in_batch_action), dim=2)

        # Repeat state to previous action states
        # observed_data = torch.cat((batch_obs.view(batch_size, 1, -1).repeat(1, batch_action.shape[1], 1),batch_action),dim=2)

        # observed_data = torch.cat((batch_obs.view(batch_size, 1, -1).repeat(1, batch_action.shape[1], 1),batch_action),dim=2)
        # observed_ts = (torch.arange(-(in_batch_action.shape[1]-1), 1, 1, device=device, dtype=torch.double) * self.dt).view(1,-1)

        # Zerofill states for previous action states
        # observed_data = torch.cat((torch.zeros(batch_size,batch_action.shape[1]-1,batch_obs.shape[1], device=device, dtype=torch.double), batch_obs.view(batch_size, 1, -1)),dim=1)
        # observed_data = torch.cat((observed_data,batch_action),dim=2)

        # tp_to_predict = torch.cat((torch.zeros(1, device=ts_pred.device, dtype=torch.double),ts_pred[0])).view(1,-1)
        data_to_predict = torch.cat((pred_batch_obs_diff.view(batch_size, 1, -1), torch.zeros((batch_size, 1, batch_action.shape[2]), device=device, dtype=torch.double)), dim=2)

        batch = {
            "observed_data": observed_data,
            "observed_tp": observed_tp,
            "data_to_predict": data_to_predict,
            "tp_to_predict": ts_pred,
            "observed_mask": torch.ones_like(observed_data),
            "mask_predicted_data": torch.ones_like(data_to_predict),
            "labels": None,
            "mode": "extrap"
        }
        loss = self.model.compute_all_losses(batch)
        return loss["loss"]

    def training_step_(self, batch):
        loss = self.model.compute_all_losses(batch)
        return loss["loss"]

    def validation_step(self, dlval):
        loss, mse = self._get_loss(dlval)
        return loss, mse

    def test_step(self, dltest):
        loss, mse = self._get_loss(dltest)
        return loss, mse

    def forward(self, in_batch_obs, in_batch_action, ts_pred):
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        # if self.normalize_time:
        #         ts_pred = (ts_pred / (self.dt*8.0))

        if len(in_batch_obs.shape) == 3:
            observed_data = torch.cat((batch_obs, batch_action), dim=2)
        else:
            if len(batch_action.shape) == 2:
                batch_action = batch_action.unsqueeze(1)

            if batch_obs.shape[0] == 1:
                self.batch_obs_buffer[0,] = torch.roll(self.batch_obs_buffer[0,], -1, dims=0)
                self.batch_obs_buffer[:,-1,:] =batch_obs
                observed_data = torch.cat((self.batch_obs_buffer, batch_action),dim=2)
            else:
                if self.batch_obs_buffer.shape[0] != batch_obs.shape[0]:
                    self.batch_obs_buffer = torch.zeros(batch_obs.shape[0],4,batch_obs.shape[1]).to(device)
                self.batch_obs_buffer = torch.roll(self.batch_obs_buffer, -1, dims=1)
                self.batch_obs_buffer[:,-1,:] = batch_obs
                observed_data = torch.cat((self.batch_obs_buffer, batch_action),dim=2)
                # observed_data = torch.cat((batch_obs.view(batch_size, 1, -1).repeat(1, batch_action.shape[1], 1), batch_action),dim=2)
        observed_ts = (torch.arange(-(in_batch_action.shape[1]-1), 1, 1, device=device, dtype=torch.double) * self.dt).view(1,-1)

        if ts_pred.shape[0] > 1:
            if ts_pred.unique().size()[0] == 1:
                ts_pred = ts_pred[0].view(1,1)
            else:
                raise ValueError("ts_pred must be unique")

        batch = {
            "observed_data": observed_data,
            "observed_tp": observed_ts,
            "data_to_predict": None,
            "tp_to_predict": ts_pred,
            "observed_mask": torch.ones_like(observed_data),
            "mask_predicted_data": None,
            "labels": None,
            "mode": "extrap"
        }
        predict = self.predict_(batch)
        return predict[:,:,:-in_batch_action.shape[2]].squeeze()

    def predict_(self, batch):
        pred_y, _ = self.model.get_reconstruction(
            batch["tp_to_predict"],
            batch["observed_data"],
            batch["observed_tp"],
            mask=batch["observed_mask"],
            n_traj_samples=1,
            mode=batch["mode"],
        )
        return pred_y.squeeze(0)

    def encode(self, dl):
        encodings = []
        for batch in dl:
            mask = batch["observed_mask"]
            truth_w_mask = batch["observed_data"]
            if mask is not None:
                truth_w_mask = torch.cat((batch["observed_data"], mask), -1)
            mean, std = self.model.encoder_z0(
                truth_w_mask, torch.flatten(batch["observed_tp"]), run_backwards=True
            )
            encodings.append(mean.view(-1, self.latents))
        return torch.cat(encodings, 0)

    def _get_and_reset_nfes(self):
        """Returns and resets the number of function evaluations for model."""
        iteration_nfes = (
            self.model.encoder_z0.z0_diffeq_solver.ode_func.nfe
            + self.model.diffeq_solver.ode_func.nfe
        )
        self.model.encoder_z0.z0_diffeq_solver.ode_func.nfe = 0
        self.model.diffeq_solver.ode_func.nfe = 0
        return iteration_nfes

def load_replay_buffer(fn):
    offline_dataset = np.load(fn, allow_pickle=True).item()
    return offline_dataset

def get_latent_ode(train_env_task,
                    delay,
                    config={},
                    model_seed=0,
                    retrain=False,
                    start_from_checkpoint=False,
                    rand=False,
                    force_retrain=False,
                    wandb=None,
                    print_settings=True):
    model_name = 'latent_ode'
    model_saved_name = f'{model_name}_{train_env_task}_delay-{delay}_ts-grid-{config.ts_grid}_{model_seed}.pt'
    model_path = f'{config.saved_models_path}{model_saved_name}'
    env = create_env(train_env_task, ts_grid=config.ts_grid, dt=config.dt * config.train_dt_multiple)
    obs_state = env.reset()

    state_dim = obs_state.shape[0]
    action_dim = env.action_space.shape[0]

    if not retrain:
        s0, a0, sn, ts = generate_irregular_data_delay(train_env_task, env, samples_per_dim=2, rand=rand, delay=delay)
    else:    
        s0, a0, sn, ts = generate_irregular_data_delay(train_env_task, env, samples_per_dim=15, rand=rand, delay=delay)

    state_mean = s0.mean(0).detach().cpu().numpy()
    state_std = s0.std(0).detach().cpu().numpy()
    action_mean = a0.mean().detach().cpu().numpy()
    ACTION_HIGH = env.action_space.high[0]
    action_std = np.array([ACTION_HIGH/2.0])

    state_diff = sn - s0
    state_diff_mean = state_diff.mean(0)
    state_diff_std = state_diff.std(0)

    latent_dim = state_dim
    hidden_units=config.latent_ode_hidden_units

    model = GeneralLatentODEOfficial(
                state_dim,
                action_dim,
                latent_dim,
                hidden_units=hidden_units,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
                normalize=config.normalize,
                dt=config.dt,
                classif_per_tp=False,
                n_labels=1,
                obsrv_std=config.latent_ode_obsrv_std,
                ).to(device)
    model.double()
    model_number_of_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f'[Model] params={model_number_of_parameters}')
    loss_func = nn.MSELoss()

    if not force_retrain:
        logger.info(f'Trying to load : {model_path}')
        if not retrain and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            return model.eval()
        if start_from_checkpoint and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))

    if print_settings:
        logger.info(f'[RUN SETTINGS]: {config}')
    if wandb is not None:
        wandb.config.update({f"{model_name}__number_of_parameters": model_number_of_parameters}, allow_val_change=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma, verbose=True)
    loss_l = []
    eval_l = []
    model.train()
    cum_loss = 0
    iters = 0

    best_loss = float("inf")
    waiting = 0
    patience = float("inf")

    batch_size = config.training_batch_size
    observed_ts = (torch.arange(-delay, 1, 1, device=device, dtype=torch.double) * config.dt).view(1,-1)
    for epoch_i in range(config.training_epochs):
        iters = 0
        cum_loss = 0
        t0 = time.perf_counter()

        as0, aa0, asn, ats = generate_irregular_data_delay(train_env_task, env, samples_per_dim=config.train_samples_per_dim, rand=rand, delay=delay)
        vs0, va0, vsn, vts = compute_val_data_delay(train_env_task, env, delay=delay, dt=config.dt, samples_per_dim=3)
        unique_times = ats.unique().shape[0]
        step = int(ats.shape[0] / unique_times)
        for j in range(unique_times):
            ts = ats[j*step: (j + 1) * step]
            s0 = as0[j*step: (j + 1) * step]
            a0 = aa0[j*step: (j + 1) * step]
            sn = asn[j*step: (j + 1) * step]
            assert ts.unique().shape[0] == 1
            s0, a0, sn, ts = s0.to(device), a0.to(device), sn.to(device), ts.to(device)
            permutation = torch.randperm(s0.size()[0])
            if int(s0.size()[0]/batch_size) < config.iters_per_log:
                config.update({'iters_per_log': int(s0.size()[0]/batch_size)}, allow_val_change=True)
            for iter_i in range(int(s0.size()[0]/batch_size)):
                optimizer.zero_grad()
                indices = permutation[iter_i*batch_size:iter_i*batch_size+batch_size]
                bs0, ba0, bsn, bts = s0[indices], a0[indices], sn[indices], ts[indices]
                bsd = bsn - bs0
                loss = model.train_step(bs0, ba0, bts, observed_ts, bsd)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()
                cum_loss += loss.item()
                iters += 1
                if iter_i > config.training_max_iters:
                    break
                if iter_i % (config.iters_per_log - 1) == 0 and not iter_i == 0:
                    val_loss = get_val_loss_delay_precomputed(model, vs0, va0, vsn, vts)
                    track_loss = cum_loss / iters
                    logger.info(f'[epoch={epoch_i+1:04d}|iter={j*step + iter_i+1:04d}/{int(s0.size()[0]/batch_size):04d}/{unique_times * step}] train_loss={track_loss:.5f} \t| val_loss={val_loss:.5f} \t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}')
                    t0 = time.perf_counter()
                    if wandb is not None:
                        wandb.log({"loss": track_loss, "epoch": epoch_i, "val_loss": val_loss, "model_name": model_name})
                    cum_loss = 0
                    iters = 0

                    # Early stopping procedure
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = deepcopy(model.state_dict())
                        torch.save(model.state_dict(), model_path)
                        waiting = 0
                    elif waiting > patience:
                        break
                    else:
                        waiting += 1
            if iter_i > config.training_max_iters:
                break
        if iter_i > config.training_max_iters:
            break
        scheduler.step()
        loss_l.append(loss.item())

    logger.info(f'[Training Finished] model: {model_name} \t|[epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/{int(s0.size()[0]/batch_size):04d}] train_loss={track_loss:.5f} \t| val_loss={val_loss:.5f} \t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}')
    model.load_state_dict(best_model)
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    results = {'val_loss': val_loss, 'train_loss': loss.item()}
    return model.eval(), results


if __name__ == '__main__':
    import wandb, sys
    defaults = get_config()
    defaults['iters_per_log'] = 10
    resume = sys.argv[-1] == "--resume"
    wandb.init(config=defaults, resume=resume, project=defaults['wandb_project'], mode="disabled")
    config = wandb.config
    logger = setup_logger(__file__)
    from config import seed_all
    seed_all(0)
    logger.info('Training a new latent_ode')
    import sys
    # logger.info(get_latent_ode('oderl-cartpole',
    #                             config=config,
    #                             retrain=True,
    #                             start_from_checkpoint=True,
    #                             delay=2,
    #                             rand=True,
    #                             force_retrain=True))
    logger.info(get_latent_ode('oderl-cartpole',
                                config=config,
                                retrain=False,
                                start_from_checkpoint=True,
                                delay=2,
                                rand=True,
                                force_retrain=False))


# Comments
# Can speed up by removing .double(), how we load data, set gradients to zero, and where any casting is done to device etc, create tensors directly