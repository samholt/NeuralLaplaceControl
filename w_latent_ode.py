import logging

import matplotlib
import matplotlib.pyplot
import torch
from torch import nn
from torchlaplace import laplace_reconstruct

from baseline_models.latent_ode_lib.create_latent_ode_model import (
    create_LatentODE_model_direct,
)
from baseline_models.latent_ode_lib.plotting import Normal
from baseline_models.latent_ode_lib.utils import compute_loss_all_batches_direct

matplotlib.use("Agg")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()


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
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
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
        self.batch_obs_buffer = torch.zeros(1, 4, state_dim).to(device)

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
        p_action = self.action_encoder(batch_action)  # pyright: ignore
        sa_in = torch.cat((batch_obs, p_action), axis=1)  # pyright: ignore
        if len(sa_in.shape) == 2:
            sa_in = sa_in.unsqueeze(1)

        p = sa_in.squeeze()
        return torch.squeeze(
            laplace_reconstruct(
                self.laplace_rep_func,
                p,
                ts_pred,
                recon_dim=self.output_dim,
                ilt_algorithm=self.ilt_algorithm,  # pyright: ignore
            )
        )

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
        data_to_predict = torch.cat(
            (
                pred_batch_obs_diff.view(batch_size, 1, -1),
                torch.zeros((batch_size, 1, batch_action.shape[2]), device=device, dtype=torch.double),
            ),
            dim=2,
        )

        batch = {
            "observed_data": observed_data,
            "observed_tp": observed_tp,
            "data_to_predict": data_to_predict,
            "tp_to_predict": ts_pred,
            "observed_mask": torch.ones_like(observed_data),
            "mask_predicted_data": torch.ones_like(data_to_predict),
            "labels": None,
            "mode": "extrap",
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
                self.batch_obs_buffer[:, -1, :] = batch_obs
                observed_data = torch.cat((self.batch_obs_buffer, batch_action), dim=2)
            else:
                if self.batch_obs_buffer.shape[0] != batch_obs.shape[0]:
                    self.batch_obs_buffer = torch.zeros(batch_obs.shape[0], 4, batch_obs.shape[1]).to(device)
                self.batch_obs_buffer = torch.roll(self.batch_obs_buffer, -1, dims=1)
                self.batch_obs_buffer[:, -1, :] = batch_obs
                observed_data = torch.cat((self.batch_obs_buffer, batch_action), dim=2)
                # observed_data = torch.cat((batch_obs.view(batch_size, 1, -1)\
                # .repeat(1, batch_action.shape[1], 1), batch_action),dim=2)
        observed_ts = (
            torch.arange(-(in_batch_action.shape[1] - 1), 1, 1, device=device, dtype=torch.double) * self.dt
        ).view(1, -1)

        if ts_pred.shape[0] > 1:
            if ts_pred.unique().size()[0] == 1:
                ts_pred = ts_pred[0].view(1, 1)
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
            "mode": "extrap",
        }
        predict = self.predict_(batch)
        return predict[:, :, : -in_batch_action.shape[2]].squeeze()

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
            # pylint: disable-next=unused-variable
            mean, std = self.model.encoder_z0(truth_w_mask, torch.flatten(batch["observed_tp"]), run_backwards=True)
            encodings.append(mean.view(-1, self.latents))
        return torch.cat(encodings, 0)

    def _get_and_reset_nfes(self):
        """Returns and resets the number of function evaluations for model."""
        iteration_nfes = (
            self.model.encoder_z0.z0_diffeq_solver.ode_func.nfe  # pyright: ignore
            + self.model.diffeq_solver.ode_func.nfe
        )
        self.model.encoder_z0.z0_diffeq_solver.ode_func.nfe = 0  # pyright: ignore
        self.model.diffeq_solver.ode_func.nfe = 0
        return iteration_nfes
