import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.multiprocessing import get_logger
from torchdiffeq import odeint

from config import get_config, seed_all
from mppi_with_model import mppi_with_model_evaluate_single_step
from overlay import (
    create_env,
    generate_irregular_data_delay_time_multi,
    load_expert_irregular_data_delay_time_multi,
    setup_logger,
)
from w_latent_ode import GeneralLatentODEOfficial
from w_nl import NeuralLaplaceModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


logger = get_logger()


def get_nl_model(
    state_dim,
    action_dim,
    state_mean,
    action_mean,
    state_std,
    action_std,
    config,  # pylint: disable=redefined-outer-name
):
    latent_dim = state_dim

    return NeuralLaplaceModel(
        state_dim,
        action_dim,
        latent_dim,
        hidden_units=config.nl_hidden_units,
        s_recon_terms=config.nl_s_recon_terms,
        ilt_algorithm=config.nl_ilt_algorithm,
        encode_obs_time=config.encode_obs_time,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        normalize=config.normalize,
        normalize_time=config.normalize_time,
    )


def get_delta_t_rnn_model(
    state_dim,
    action_dim,
    state_mean,
    action_mean,
    state_std,
    action_std,
    config,  # pylint: disable=redefined-outer-name
):
    return DeltaTRNN(
        state_dim,
        action_dim,
        hidden_units=config.rnn_hidden_units,
        encode_obs_time=config.encode_obs_time,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        normalize=config.normalize,
        normalize_time=config.normalize_time,
    )


def get_rnn_model(
    state_dim,
    action_dim,
    state_mean,
    action_mean,
    state_std,
    action_std,
    config,  # pylint: disable=redefined-outer-name
):
    return RNN(
        state_dim,
        action_dim,
        hidden_units=config.rnn_hidden_units,
        encode_obs_time=config.encode_obs_time,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        normalize=config.normalize,
    )


def get_node_model(
    state_dim,
    action_dim,
    state_mean,
    action_mean,
    state_std,
    action_std,
    config,  # pylint: disable=redefined-outer-name
):
    latent_dim = state_dim
    return NODE(
        state_dim,
        action_dim,
        latent_dim,
        hidden_units=config.node_hidden_units,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        normalize=config.normalize,
        normalize_time=config.normalize_time,
        encode_obs_time=False,
        method=config.node_method,
        augment_dim=config.node_augment_dim,
    )


def get_latent_ode_model(
    state_dim,
    action_dim,
    state_mean,
    action_mean,
    state_std,
    action_std,
    config,  # pylint: disable=redefined-outer-name
):
    latent_dim = state_dim

    # pylint: disable-next=unexpected-keyword-arg,redundant-keyword-arg
    return GeneralLatentODEOfficial(
        state_dim,
        action_dim,
        latent_dim,
        hidden_units=config.latent_ode_hidden_units,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        normalize=config.normalize,
        normalize_time=config.normalize_time,
        dt=config.dt,
        classif_per_tp=False,
        n_labels=1,
        obsrv_std=config.latent_ode_obsrv_std,
    )


def train_model(
    model_name,  # pylint: disable=redefined-outer-name
    train_env_task,  # pylint: disable=redefined-outer-name
    config,  # pylint: disable=redefined-outer-name
    wandb,  # pylint: disable=redefined-outer-name
    delay,
    retrain=False,
    force_retrain=False,
    model_seed=0,
    start_from_checkpoint=False,
    print_settings=True,
    evaluate_model_when_trained=False,
):
    model_saved_name = (
        f"{model_name}_{train_env_task}_delay-{delay}_ts-grid-{config.ts_grid}_"
        f"{model_seed}_train-with-expert-trajectories-{config.train_with_expert_trajectories}"
    )
    if config.end_training_after_seconds is None:
        model_saved_name = f"{model_saved_name}_training_for_epochs-{config.training_epochs}"
    if config.training_use_only_samples is not None:
        model_saved_name = f"{model_saved_name}_samples_used-{config.training_use_only_samples}"
    model_saved_name = f"{model_saved_name}.pt"
    model_path = f"{config.saved_models_path}{model_saved_name}"
    env = create_env(train_env_task, ts_grid=config.ts_grid, dt=config.dt * config.train_dt_multiple)
    obs_state = env.reset()
    state_dim = obs_state.shape[0]
    action_dim = env.action_space.shape[0]  # pyright: ignore

    action_mean = np.array([0] * action_dim)
    ACTION_HIGH = env.action_space.high[0]  # pyright: ignore
    if train_env_task == "oderl-cartpole":
        state_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        state_std = np.array([2.88646771, 11.54556671, 0.70729307, 0.70692035, 17.3199048])
        action_std = np.array([ACTION_HIGH / 2.0])
    elif train_env_task == "oderl-pendulum":
        state_mean = np.array([0.0, 0.0, 0.0])
        state_std = np.array([0.70634571, 0.70784512, 2.89072771])
        action_std = np.array([ACTION_HIGH / 2.0])
    elif train_env_task == "oderl-acrobot":
        state_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_std = np.array([0.70711024, 0.70710328, 0.7072186, 0.7069949, 2.88642115, 2.88627309])
        action_std = np.array([ACTION_HIGH / 2.0])

    action_mean = np.array([0] * action_dim)
    ACTION_HIGH = env.action_space.high[0]  # pyright: ignore
    if train_env_task == "oderl-cartpole":
        state_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        state_std = np.array([2.88646771, 11.54556671, 0.70729307, 0.70692035, 17.3199048])
        action_std = np.array([ACTION_HIGH / 2.0])
    elif train_env_task == "oderl-pendulum":
        state_mean = np.array([0.0, 0.0, 0.0])
        state_std = np.array([0.70634571, 0.70784512, 2.89072771])
        action_std = np.array([ACTION_HIGH / 2.0])
    elif train_env_task == "oderl-acrobot":
        state_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_std = np.array([0.70711024, 0.70710328, 0.7072186, 0.7069949, 2.88642115, 2.88627309])
        action_std = np.array([ACTION_HIGH / 2.0])

    if model_name == "nl":
        model = get_nl_model(
            state_dim,
            action_dim,
            state_mean,  # pyright: ignore
            action_mean,
            state_std,  # pyright: ignore
            action_std,  # pyright: ignore
            config,
        ).to(device)
    elif model_name == "delta_t_rnn":
        model = get_delta_t_rnn_model(
            state_dim,
            action_dim,
            state_mean,  # pyright: ignore
            action_mean,
            state_std,  # pyright: ignore
            action_std,  # pyright: ignore
            config,
        ).to(device)
    elif model_name == "rnn":
        model = get_rnn_model(
            state_dim,
            action_dim,
            state_mean,  # pyright: ignore
            action_mean,
            state_std,  # pyright: ignore
            action_std,  # pyright: ignore
            config,
        ).to(device)
    elif model_name == "latent_ode":
        model = get_latent_ode_model(
            state_dim,
            action_dim,
            state_mean,  # pyright: ignore
            action_mean,
            state_std,  # pyright: ignore
            action_std,  # pyright: ignore
            config,
        ).to(device)
    elif model_name == "node":
        model = get_node_model(
            state_dim,
            action_dim,
            state_mean,  # pyright: ignore
            action_mean,
            state_std,  # pyright: ignore
            action_std,  # pyright: ignore
            config,
        ).to(device)
    model.double()  # pyright: ignore
    model_number_of_parameters = sum(p.numel() for p in model.parameters())  # pyright: ignore
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(  # pyright: ignore
        f"[{train_env_task}\t{model_name}\td={delay}\tsamples={config.training_use_only_samples}][Model] "
        f"params={model_number_of_parameters}"
    )
    loss_func = nn.MSELoss()

    if not force_retrain:
        # pylint: disable-next=logging-fstring-interpolation
        logger.info(  # pyright: ignore
            f"[{train_env_task}\t{model_name}\td={delay}\tsamples={config.training_use_only_samples}]"
            f"Trying to load : {model_path}"
        )
        if not retrain and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))  # pyright: ignore
            return model.eval(), {"total_reward": None}  # pyright: ignore
        elif not retrain:
            raise ValueError
        if start_from_checkpoint and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))  # pyright: ignore
    if print_settings:
        # pylint: disable-next=logging-fstring-interpolation
        logger.info(  # pyright: ignore
            f"[{train_env_task}\t{model_name}\td={delay}\tsamples={config.training_use_only_samples}]"
            f"[RUN SETTINGS]: {config}"
        )
    if wandb is not None:
        wandb.config.update({f"{model_name}__number_of_parameters": model_number_of_parameters}, allow_val_change=True)
    optimizer = optim.Adam(
        model.parameters(),  # pyright: ignore
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_scheduler_step_size,
            gamma=config.lr_scheduler_gamma,
            verbose=True,
        )
    loss_l = []
    # eval_l = []
    model.train()  # pyright: ignore
    cum_loss = 0
    iters = 0

    best_loss = float("inf")
    waiting = 0
    patience = float("inf")

    if "ode" in model_name:
        batch_size = 1  # NODE methods do not support bach trajectories of different times.
    else:
        batch_size = config.training_batch_size

    if model_name == "latent_ode":
        observed_ts = (
            torch.arange(
                -(config.action_buffer_size - 1),
                1,
                1,
                device=device,
                dtype=torch.double,
            )
            * config.dt
        ).view(1, -1)
    train_start_time = time.perf_counter()
    # elapsed_time_loss_l = []
    # best_model = None
    elapsed_time = time.perf_counter() - train_start_time
    if config.train_with_expert_trajectories and config.training_use_only_samples is not None:
        s0, a0, sn, ts = load_expert_irregular_data_delay_time_multi(
            train_env_task, delay, encode_obs_time=config.encode_obs_time, config=config
        )
        permutation = torch.randperm(s0.size()[0])
        permutation = permutation[: config.training_use_only_samples]
    for epoch_i in range(config.training_epochs):
        iters = 0
        cum_loss = 0
        t0 = time.perf_counter()
        # if model_name == "latent_ode":
        #     samples_per_dim = 3
        # else:
        #     samples_per_dim = config.train_samples_per_dim
        if config.train_with_expert_trajectories:
            s0, a0, sn, ts = load_expert_irregular_data_delay_time_multi(
                train_env_task,
                delay,
                encode_obs_time=config.encode_obs_time,
                config=config,
            )
        else:
            s0, a0, sn, ts = generate_irregular_data_delay_time_multi(
                train_env_task,
                env,
                samples_per_dim=config.train_samples_per_dim,
                rand=config.rand_sample,
                delay=delay,
                encode_obs_time=config.encode_obs_time,
                action_buffer_size=config.action_buffer_size,
                reuse_state_actions_when_sampling_times=config.reuse_state_actions_when_sampling_times,
            )
        if "latent_ode" in model_name:
            current_a0 = a0[:, -1, :]
            history_s0 = s0.unfold(dimension=0, size=config.action_buffer_size, step=1).permute(0, 2, 1)
            history_a0 = current_a0.unfold(dimension=0, size=config.action_buffer_size, step=1).permute(0, 2, 1)
            sn, ts = sn[: -(config.action_buffer_size - 1)].to(device), ts[: -(config.action_buffer_size - 1)].to(
                device
            )
            history_s0, history_a0 = history_s0.to(device), history_a0.to(device)
        else:
            s0, a0, sn, ts = s0.to(device), a0.to(device), sn.to(device), ts.to(device)
        if config.training_use_only_samples is None:
            permutation = torch.randperm(s0.size()[0])
        if int(permutation.size()[0] / batch_size) < config.iters_per_log:  # pyright: ignore
            config.update(
                {"iters_per_log": int(permutation.size()[0] / batch_size)},  # pyright: ignore
                allow_val_change=True,
            )
        for iter_i in range(int(permutation.size()[0] / batch_size)):  # pyright: ignore
            optimizer.zero_grad()
            indices = permutation[iter_i * batch_size : iter_i * batch_size + batch_size]  # pyright: ignore
            if "latent_ode" in model_name:
                bhistory_s0, bhistory_a0, bsn, bts = (
                    history_s0[indices],  # pyright: ignore
                    history_a0[indices],  # pyright: ignore
                    sn[indices],
                    ts[indices],
                )
                bsd = bsn - bhistory_s0[:, -1, :]
                loss = model.train_step(bhistory_s0, bhistory_a0, bts, observed_ts, bsd)  # pyright: ignore
            else:
                bs0, ba0, bsn, bts = s0[indices], a0[indices], sn[indices], ts[indices]
                bsd = bsn - bs0
                pred_sd = model(bs0, ba0, bts)  # pyright: ignore  # pylint: disable=not-callable
                loss = loss_func(pred_sd.squeeze(), bsd.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)  # pyright: ignore
            optimizer.step()
            cum_loss += loss.item()
            iters += 1
            if (permutation.shape[0] == batch_size) or (  # pyright: ignore
                iter_i % (config.iters_per_log - 1) == 0 and not iter_i == 0
            ):
                track_loss = cum_loss / iters
                elapsed_time = time.perf_counter() - train_start_time
                if (
                    config.sweep_mode
                    and config.end_training_after_seconds is not None
                    and elapsed_time > config.end_training_after_seconds
                ):
                    # pylint: disable-next=logging-fstring-interpolation
                    logger.info(  # pyright: ignore
                        f"[{train_env_task}\t{model_name}\td={delay}\t"
                        f"samples={config.training_use_only_samples}]Ending training"
                    )
                    break
                # pylint: disable-next=logging-fstring-interpolation
                logger.info(  # pyright: ignore
                    f"[{train_env_task}\t{model_name}\td={delay}\tsamples={config.training_use_only_samples}]"
                    f"[epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/"
                    f"{int(permutation.size()[0]/batch_size):04d}|"  # pyright: ignore
                    f"t:{int(elapsed_time)}/{config.end_training_after_seconds if config.sweep_mode else 0}] "
                    f"train_loss={track_loss} \t\t| s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}"
                )
                t0 = time.perf_counter()
                if wandb is not None:
                    wandb.log({"loss": track_loss, "epoch": epoch_i, "model_name": model_name})
                iters = 0

                # Early stopping procedure
                if cum_loss < best_loss:
                    best_loss = cum_loss
                    torch.save(model.state_dict(), model_path)  # pyright: ignore
                    waiting = 0
                elif waiting > patience:
                    break
                else:
                    waiting += 1
                cum_loss = 0

            if iter_i % (config.iters_per_evaluation - 1) == 0 and not iter_i == 0:
                evaluate_model(
                    model,  # pyright: ignore
                    model_name,
                    train_env_task,
                    wandb,
                    config,
                    delay,
                    intermediate_run=True,
                )
        if (
            config.sweep_mode
            and config.end_training_after_seconds is not None
            and elapsed_time > config.end_training_after_seconds
        ):
            break
        if config.use_lr_scheduler:
            scheduler.step()  # pyright: ignore
        loss_l.append(loss.item())  # pyright: ignore

    # pylint: disable-next=logging-fstring-interpolation
    logger.info(  # pyright: ignore
        f"[{train_env_task}\t{model_name}\td={delay}\tsamples={config.training_use_only_samples}][Training Finished] "
        f"model: {model_name} \t|[epoch={epoch_i+1:04d}|iter={iter_i+1:04d}/"  # pyright: ignore
        f"{int(permutation.size()[0]/batch_size):04d}] train_loss={track_loss:.5f} \t| "  # pyright: ignore
        f"s/it={(time.perf_counter() - t0)/config.iters_per_log:.5f}"  # pyright: ignore
    )
    if evaluate_model_when_trained:
        total_reward = evaluate_model(
            model,  # pyright: ignore
            model_name,
            train_env_task,
            wandb,
            config,
            delay,
            intermediate_run=False,
        )
    else:
        total_reward = None
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), model_path)  # pyright: ignore
    results = {"train_loss": loss.item(), "best_val_loss": best_loss, "total_reward": total_reward}  # pyright: ignore
    return model.eval(), results  # pyright: ignore


def evaluate_model(
    model,
    model_name,  # pylint: disable=redefined-outer-name
    train_env_task,  # pylint: disable=redefined-outer-name
    wandb,  # pylint: disable=redefined-outer-name
    config,  # pylint: disable=redefined-outer-name
    delay,
    intermediate_run=False,
):
    if config.sweep_mode and not intermediate_run:
        seed_all(0)

    eval_result = mppi_with_model_evaluate_single_step(
        model_name=model_name,
        env_name=train_env_task,
        roll_outs=config.mppi_roll_outs,
        time_steps=config.mppi_time_steps,
        lambda_=config.mppi_lambda,
        sigma=config.mppi_sigma,
        dt=config.dt,
        action_delay=delay,
        encode_obs_time=config.encode_obs_time,
        config=config,
        model=model,
        # save_video=config.save_video,
        save_video=False,
        intermediate_run=intermediate_run,
    )
    total_reward = eval_result["total_reward"]
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(f"[Evaluation Result] Total reward {total_reward}")  # pyright: ignore
    if wandb is not None:
        wandb.log({"total_reward": total_reward})
    return total_reward


# def seed_all(seed=None):
#     """
#     Set the torch, numpy, and random module seeds based on the seed
#     specified in config. If there is no seed or it is None, a time-based
#     seed is used instead and is written to config.
#     """
#     # Default uses current time in milliseconds, modulo 1e9
#     if seed is None:
#         seed = round(time() * 1000) % int(1e9)  # pyright: ignore

#     # Set the seeds using the shifted seed
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     # pylint: disable-next=logging-fstring-interpolation
#     logger.info(f"[Seed] {seed}")  # pyright: ignore


# Models


class RNN(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_units=64,
        encode_obs_time=False,
        state_mean=None,
        state_std=None,
        action_mean=None,
        action_std=None,
        normalize=False,
    ):
        super(RNN, self).__init__()
        dimension_in = action_dim
        self.encode_obs_time = encode_obs_time
        latent_dim = hidden_units
        self.gru = nn.GRU(dimension_in, latent_dim, batch_first=True)
        penultimate_layer_dim = latent_dim + state_dim
        self.linear_out = nn.Linear(penultimate_layer_dim, state_dim)
        self.normalize = normalize
        self.register_buffer("state_mean", torch.tensor(state_mean))
        self.register_buffer("state_std", torch.tensor(state_std))
        self.register_buffer("action_mean", torch.tensor(action_mean))
        self.register_buffer("action_std", torch.tensor(action_std))

    def forward(self, in_batch_obs, in_batch_action, _):
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        out, _ = self.gru(batch_action)
        return self.linear_out(torch.cat((out[:, -1, :], batch_obs), dim=1))


class DeltaTRNN(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_units=64,
        encode_obs_time=False,
        state_mean=None,
        state_std=None,
        action_mean=None,
        action_std=None,
        normalize=False,
        normalize_time=False,
        dt=0.05,
    ):
        super(DeltaTRNN, self).__init__()
        dimension_in = action_dim
        if encode_obs_time:
            dimension_in += 1
        self.encode_obs_time = encode_obs_time
        latent_dim = hidden_units
        self.gru = nn.GRU(dimension_in, latent_dim, batch_first=True)
        penultimate_layer_dim = latent_dim + state_dim + 1  # For delta t
        self.linear_out = nn.Linear(penultimate_layer_dim, state_dim)
        self.normalize = normalize
        self.normalize_time = normalize_time
        self.register_buffer("state_mean", torch.tensor(state_mean))
        self.register_buffer("state_std", torch.tensor(state_std))
        self.register_buffer("action_mean", torch.tensor(action_mean))
        self.register_buffer("action_std", torch.tensor(action_std))
        self.register_buffer("dt", torch.tensor(dt))

    def forward(self, in_batch_obs, in_batch_action, ts_pred):
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            batch_action = (in_batch_action - self.action_mean) / self.action_std
        if self.normalize_time:
            ts_pred = ts_pred / (self.dt * 8.0)  # pyright: ignore
        else:
            batch_obs = in_batch_obs
            batch_action = in_batch_action / 3.0
        out, _ = self.gru(batch_action)  # pyright: ignore
        return self.linear_out(torch.cat((out[:, -1, :], batch_obs, ts_pred), dim=1))  # pyright: ignore


# NODE


class xOdeFuncInXAndU(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, nhidden=50, augment_dim=0):
        super(xOdeFuncInXAndU, self).__init__()

        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(state_dim + action_dim + augment_dim, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, state_dim + augment_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        # self.nfe = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.augment_dim = augment_dim
        self.u = None

    def update_u(self, u):
        self.u = u

    def forward(self, t, x):
        # self.nfe += 1
        return self.linear_tanh_stack(torch.cat((x, self.u), 1))  # pyright: ignore


def load_replay_buffer(fn):
    offline_dataset = np.load(fn, allow_pickle=True).item()
    return offline_dataset


class NODE(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        hidden_units=64,
        encode_obs_time=False,
        state_mean=None,
        state_std=None,
        action_mean=None,
        action_std=None,
        normalize=False,
        normalize_time=False,
        method="euler",
        augment_dim=0,
        action_high=1.0,
        dt=0.05,
    ):
        super(NODE, self).__init__()
        self.x_ode_func_in_x_and_u = xOdeFuncInXAndU(
            state_dim=state_dim,
            action_dim=action_dim,
            augment_dim=augment_dim,
            nhidden=hidden_units,
        )
        self.method = method
        self.augment_dim = augment_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.normalize = normalize
        self.encode_obs_time = encode_obs_time
        self.normalize_time = normalize_time
        self.register_buffer("state_mean", torch.tensor(state_mean))
        self.register_buffer("state_std", torch.tensor(state_std))
        self.register_buffer("action_mean", torch.tensor(action_mean))
        self.register_buffer("action_std", torch.tensor(action_std))
        self.register_buffer("dt", torch.tensor(dt))

    def forward(self, in_batch_obs, in_batch_action, ts_pred):
        if self.normalize:
            batch_obs = (in_batch_obs - self.state_mean) / self.state_std
            # batch_action = (in_batch_action - self.action_mean) / self.action_std
        else:
            batch_obs = in_batch_obs
            # batch_action = in_batch_action / self.action_high
        if self.normalize_time:
            ts_pred = ts_pred / (self.dt * 8.0)  # pyright: ignore
        # x = batch_obs[:,-1,:]
        x = batch_obs
        if self.augment_dim > 0:
            # Add augmentation
            aug = torch.zeros(batch_obs.shape[0], self.augment_dim).to(device)
            # Shape (batch_size, data_dim + augment_dim)
            x = torch.cat([x, aug], 1)

        if len(in_batch_action.shape) == 2:
            in_batch_action = in_batch_action.unsqueeze(1)

        self.x_ode_func_in_x_and_u.update_u(in_batch_action[:, -1, :])
        out = odeint(
            self.x_ode_func_in_x_and_u,
            x,
            torch.cat((torch.zeros(1, device=ts_pred.device, dtype=torch.double), ts_pred[0])),
            method=self.method,
            options={"step_size": 0.05, "norm": "seminorm"},
        )
        return out[-1, :, : out.shape[-1] - self.augment_dim]  # pyright: ignore


if __name__ == "__main__":
    defaults = get_config()
    wandb.init(config=defaults, project=defaults["wandb_project"])  # pyright: ignore
    # , mode="disabled")
    config = wandb.config
    logger = setup_logger(__file__, log_folder=config.log_folder)

    seed_all(0)
    logger.info("Training a model")  # pyright: ignore
    model_name = "latent_ode"  # 'node' #'delta_t_rnn' # 'nl'
    train_env_task = "oderl-cartpole"
    train_model(  # pyright: ignore  # pylint: disable=no-value-for-parameter
        model_name,
        train_env_task,
        config,
        wandb,
        retrain=True,
        force_retrain=True,
        model_seed=0,
        start_from_checkpoint=True,
        print_settings=True,
    )
    logger.info()  # pyright: ignore
    wandb.finish()
