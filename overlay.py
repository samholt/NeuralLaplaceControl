# General code that should be globally run and or useful re-usable utility functions
import gym
import numpy as np
import pyvirtualdisplay
import torch
import torch.nn as nn
from tqdm import tqdm

from envs.oderl import envs  # pylint: disable=import-error

from .mppi_dataset_collector import mppi_with_model_collect_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cpu'
torch.set_default_dtype(torch.float32)


def start_virtual_display():
    return pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()  # pyright: ignore


def create_oderl_env(env_name, dt=0.05, ts_grid="fixed", noise=0.0, friction=False):
    ################## environment and dataset ##################
    # dt      = 0.05 		# mean time difference between observations
    # noise   = 0.0 		# observation noise std
    # ts_grid = 'fixed' 	# the distribution for the observation time differences: ['fixed','uniform','exp']
    # ENV_CLS = envs.CTCartpole # [CTPendulum, CTCartpole, CTAcrobot]
    if env_name == "oderl-pendulum":
        ENV_CLS = envs.CTPendulum  # [CTPendulum, CTCartpole, CTAcrobot]
    elif env_name == "oderl-cartpole":
        ENV_CLS = envs.CTCartpole  # [CTPendulum, CTCartpole, CTAcrobot]
    elif env_name == "oderl-acrobot":
        ENV_CLS = envs.CTAcrobot  # [CTPendulum, CTCartpole, CTAcrobot]
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    env = ENV_CLS(
        dt=dt,
        obs_trans=True,
        device=device,  # pyright: ignore
        obs_noise=noise,
        ts_grid=ts_grid,
        solver="euler",
        friction=friction,
    )
    return env


def create_env(env_name, dt=0.05, ts_grid="fixed", noise=0.0, friction=False):
    if "oderl" in env_name:
        env = create_oderl_env(env_name, dt=dt, ts_grid=ts_grid, noise=noise, friction=friction)
    else:
        env = gym.make(env_name)
    return env


def setup_logger(file, log_folder="logs", return_path_to_log=False):
    import logging
    import os
    import time

    file_name = os.path.basename(os.path.realpath(file)).split(".py")[0]
    from pathlib import Path

    Path(f"./{log_folder}").mkdir(parents=True, exist_ok=True)
    path_run_name = "{}-{}".format(file_name, time.strftime("%Y%m%d-%H%M%S"))
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(f"{log_folder}/{path_run_name}_log.txt"),
            logging.StreamHandler(),
        ],
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    # pylint: disable-next=logging-fstring-interpolation
    logger.info(f"Starting: Log file at: {log_folder}/{path_run_name}_log.txt")
    if return_path_to_log:
        return logger, f"{log_folder}/{path_run_name}_log.txt"
    else:
        return logger


def load_replay_buffer(fn):
    offline_dataset = np.load(fn, allow_pickle=True).item()
    return offline_dataset


def get_val_loss_delay_latent(model, train_env_task, env, delay=0, dt=0.05):
    s0, a0, sb, sn, _ = generate_irregular_data_delay_latent(
        train_env_task, env, samples_per_dim=5, delay=delay, latent=True
    )
    s0, a0, sb, sn = s0.to(device), a0.to(device), sb.to(device), sn.to(device)
    ts = torch.tensor([0.05]).to(device).view(1, 1).repeat(s0.shape[0], 1)
    s0 = s0.double()
    sb = sb.double()
    a0 = a0.double()
    sn = sn.double()
    ts = ts.double()
    pred_sd = model(
        torch.cat(
            (
                s0.view(s0.shape[0], 1, s0.shape[1]),
                sb.view(sb.shape[0], 1, sb.shape[1]),
            ),
            dim=1,
        ),
        a0,
        ts,
    )
    sd = sn - s0
    return nn.MSELoss()(pred_sd, sd).item()


def get_val_loss_delay_precomputed(model, s0, a0, sn, ts):
    pred_sd = model(s0, a0, ts)
    sd = sn - s0
    return nn.MSELoss()(pred_sd.squeeze(), sd.squeeze()).item()


def compute_val_data_delay(train_env_task, env, delay, dt=0.05, samples_per_dim=5):
    s0, a0, sn, _ = generate_irregular_data_delay(train_env_task, env, samples_per_dim=samples_per_dim, delay=delay)
    s0, a0, sn = s0.to(device), a0.to(device), sn.to(device)
    ts = torch.tensor([0.05]).to(device).view(1, 1).repeat(s0.shape[0], 1)
    if train_env_task == "oderl-cartpole":
        from .oracle import cartpole_dynamics_dt_delay

        sn = cartpole_dynamics_dt_delay(s0, a0, ts, delay=delay)
    elif train_env_task == "oderl-pendulum":
        from .oracle import pendulum_dynamics_dt_delay

        sn = pendulum_dynamics_dt_delay(s0, a0, ts, delay=delay)
    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()  # pyright: ignore
    ts = ts.double()
    return s0, a0, sn, ts


def get_val_loss_delay_time_multi(
    model,
    train_env_task,
    env,
    delay,
    dt=0.05,
    samples_per_dim=5,
    encode_obs_time=False,
    action_buffer_size=5,
):
    s0, a0, sn, _ = generate_irregular_data_delay_time_multi(
        train_env_task,
        env,
        samples_per_dim=samples_per_dim,
        delay=delay,
        encode_obs_time=encode_obs_time,
        action_buffer_size=action_buffer_size,
    )
    s0, a0, sn = s0.to(device), a0.to(device), sn.to(device)
    ts = torch.tensor([0.05]).to(device).view(1, 1).repeat(s0.shape[0], 1)
    if train_env_task == "oderl-cartpole":
        from .oracle import cartpole_dynamics_dt_delay

        if encode_obs_time:
            sn = cartpole_dynamics_dt_delay(s0, a0[:, :, :1], ts, delay=delay)
        else:
            sn = cartpole_dynamics_dt_delay(s0, a0, ts, delay=delay)
    elif train_env_task == "oderl-pendulum":
        from .oracle import pendulum_dynamics_dt_delay

        if encode_obs_time:
            sn = pendulum_dynamics_dt_delay(s0, a0[:, :, :1], ts, delay=delay)
        else:
            sn = pendulum_dynamics_dt_delay(s0, a0, ts, delay=delay)
    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()  # pyright: ignore
    ts = ts.double()
    pred_sd = model(s0, a0, ts)
    sd = sn - s0
    return nn.MSELoss()(pred_sd.squeeze(), sd.squeeze()).item()


def get_val_loss_delay(model, train_env_task, env, delay, dt=0.05, samples_per_dim=5):
    s0, a0, sn, _ = generate_irregular_data_delay(train_env_task, env, samples_per_dim=samples_per_dim, delay=delay)
    s0, a0, sn = s0.to(device), a0.to(device), sn.to(device)
    ts = torch.tensor([0.05]).to(device).view(1, 1).repeat(s0.shape[0], 1)
    if train_env_task == "oderl-cartpole":
        from .oracle import cartpole_dynamics_dt_delay

        sn = cartpole_dynamics_dt_delay(s0, a0, ts, delay=delay)
    elif train_env_task == "oderl-pendulum":
        from .oracle import pendulum_dynamics_dt_delay

        sn = pendulum_dynamics_dt_delay(s0, a0, ts, delay=delay)
    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()  # pyright: ignore
    ts = ts.double()
    pred_sd = model(s0, a0, ts)
    sd = sn - s0
    return nn.MSELoss()(pred_sd, sd).item()


def get_val_loss(model, train_env_task, env):
    s0, a0, sn, ts = generate_irregular_data(train_env_task, env, samples_per_dim=5)
    s0, a0, sn, ts = s0.to(device), a0.to(device), sn.to(device), ts.to(device)
    if train_env_task == "oderl-cartpole":
        raise NotImplementedError
        # from .oracle import cartpole_dynamics
        # sn = cartpole_dynamics(s0, a0)
    elif train_env_task == "oderl-pendulum":
        from .oracle import pendulum_dynamics_dt

        sn = pendulum_dynamics_dt(s0, a0)  # pyright: ignore  # pylint: disable=no-value-for-parameter
    ts = torch.tensor([0.05]).to(device).view(1, 1).repeat(s0.shape[0], 1)
    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()
    ts = ts.double()
    pred_sd = model(s0, a0, ts)
    sd = sn - s0
    return nn.MSELoss()(pred_sd, sd).item()


def generate_irregular_data_delay_latent(
    train_env_task,
    env,
    delay,
    samples_per_dim=None,
    mode="grid",
    rand=False,
    latent=False,
):
    if samples_per_dim is None:
        if train_env_task == "oderl-pendulum":
            samples_per_dim = 33
        elif train_env_task == "oderl-cartpole":
            samples_per_dim = 20
        elif train_env_task == "oderl-acrobot":
            samples_per_dim = 15

    ACTION_HIGH = env.action_space.high[0]
    # ACTION_LOW = env.action_space.low[0]
    nu = env.action_space.shape[0]
    if train_env_task == "oderl-cartpole":
        state_max = torch.tensor([5.0, 20, torch.pi, 30])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sb_l, sn_l, ts_l = [], [], [], [], []
        # pylint: disable-next=unused-variable
        for ti in range(samples_per_dim):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**4, 4, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim, 1, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
                actions = actions.view(-1)
            else:
                x, y, z, k = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[2], state_max[2], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[3], state_max[3], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat(
                    (
                        x.unsqueeze(-1),
                        y.unsqueeze(-1),
                        z.unsqueeze(-1),
                        k.unsqueeze(-1),
                    ),
                    -1,
                )
                s0s = all_t.view(-1, 4)
                actions = torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h)
            s0, a0, sb, sn, ts = env.batch_integrate_system_double_time(
                s0s, actions, device=device_h
            )  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sb_l.append(sb), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore
    elif train_env_task == "oderl-pendulum":
        state_max = torch.tensor([torch.pi, 5.0])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        for ti in range(samples_per_dim):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**2, 2, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim, 1, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
                actions = actions.view(-1)
            else:
                x, y = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), -1)
                s0s = all_t.view(-1, 2)
                actions = torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h)
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h)  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore
    elif train_env_task == "oderl-acrobot":
        state_max = torch.tensor([torch.pi, torch.pi, 5.0, 5.0])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        for ti in tqdm(range(samples_per_dim)):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**4, 4, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([ACTION_HIGH, ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim**2, 2, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
            else:
                x, y, z, k = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[2], state_max[2], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[3], state_max[3], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat(
                    (
                        x.unsqueeze(-1),
                        y.unsqueeze(-1),
                        z.unsqueeze(-1),
                        k.unsqueeze(-1),
                    ),
                    -1,
                )
                s0s = all_t.view(-1, 4)
                if env.action_space.shape[0] == 2:
                    a1, a2 = torch.meshgrid(
                        torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h),
                        torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h),
                        indexing="ij",
                    )
                    a_t = torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1)
                    actions = a_t.view(-1, 2)
            s0, a0, sn, ts = env.batch_integrate_system(
                s0s, actions, device=device_h  # pyright: ignore
            )  # Stochastic sampling of time
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore

    s0 = torch.cat(s0_l, dim=0)  # pyright: ignore
    a0 = torch.cat(a0_l, dim=0)  # pyright: ignore
    sb = torch.cat(sb_l, dim=0)  # pyright: ignore
    sn = torch.cat(sn_l, dim=0)  # pyright: ignore
    ts = torch.cat(ts_l, dim=0)  # pyright: ignore

    if delay > 0:
        a = (
            (torch.rand(a0.shape[0], delay, nu, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
            * 2.0
            * ACTION_HIGH
        )
        a0 = torch.cat((a0.view(a0.shape[0], -1, nu), a), dim=1)

    if latent:
        from .oracle import cartpole_dynamics_dt_latent

        sn = cartpole_dynamics_dt_latent(sb, s0, a0, ts)
        s0 = s0[:, [0, 2, 3]]  # Remove x_dot, theta_dot
        sb = sb[:, [0, 2, 3]]  # Remove x_dot, theta_dot
        sn = sn[:, [0, 2, 3]]  # Remove x_dot, theta_dot

    s0 = s0.double()
    sb = sb.double()
    a0 = a0.double()
    sn = sn.double()
    ts = ts.double()
    return s0.detach(), a0.detach(), sb.detach(), sn.detach(), ts.detach()


def generate_irregular_data_delay(train_env_task, env, delay, samples_per_dim=None, mode="grid", rand=False):
    if samples_per_dim is None:
        if train_env_task == "oderl-pendulum":
            samples_per_dim = 33
        elif train_env_task == "oderl-cartpole":
            samples_per_dim = 20
        elif train_env_task == "oderl-acrobot":
            samples_per_dim = 15

    time_multiplier = 10
    ACTION_HIGH = env.action_space.high[0]
    # ACTION_LOW = env.action_space.low[0]
    nu = env.action_space.shape[0]
    if train_env_task == "oderl-cartpole":
        state_max = torch.tensor([5.0, 20, torch.pi, 30])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        # pylint: disable-next=unused-variable
        for ti in range(int(samples_per_dim * time_multiplier)):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**4, 4, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim, 1, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
                actions = actions.view(-1)
            else:
                x, y, z, k = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[2], state_max[2], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[3], state_max[3], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat(
                    (
                        x.unsqueeze(-1),
                        y.unsqueeze(-1),
                        z.unsqueeze(-1),
                        k.unsqueeze(-1),
                    ),
                    -1,
                )
                s0s = all_t.view(-1, 4)
                actions = torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h)
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h)  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore
    elif train_env_task == "oderl-pendulum":
        state_max = torch.tensor([torch.pi, 5.0])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        for ti in range(samples_per_dim):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**2, 2, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim, 1, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
                actions = actions.view(-1)
            else:
                x, y = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), -1)
                s0s = all_t.view(-1, 2)
                actions = torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h)
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h)  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore
    elif train_env_task == "oderl-acrobot":
        state_max = torch.tensor([torch.pi, torch.pi, 5.0, 5.0])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        for ti in tqdm(range(samples_per_dim)):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**4, 4, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([ACTION_HIGH, ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim**2, 2, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
            else:
                x, y, z, k = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[2], state_max[2], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[3], state_max[3], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat(
                    (
                        x.unsqueeze(-1),
                        y.unsqueeze(-1),
                        z.unsqueeze(-1),
                        k.unsqueeze(-1),
                    ),
                    -1,
                )
                s0s = all_t.view(-1, 4)
                if env.action_space.shape[0] == 2:
                    a1, a2 = torch.meshgrid(
                        torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h),
                        torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h),
                        indexing="ij",
                    )
                    a_t = torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1)
                    actions = a_t.view(-1, 2)
            s0, a0, sn, ts = env.batch_integrate_system(
                s0s, actions, device=device_h  # pyright: ignore
            )  # Stochastic sampling of time
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore

    s0 = torch.cat(s0_l, dim=0)  # pyright: ignore
    a0 = torch.cat(a0_l, dim=0)  # pyright: ignore
    sn = torch.cat(sn_l, dim=0)  # pyright: ignore
    ts = torch.cat(ts_l, dim=0)  # pyright: ignore

    if delay > 0:
        a = (
            (torch.rand(a0.shape[0], delay, nu, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
            * 2.0
            * ACTION_HIGH
        )
        # a = torch.zeros_like(a)
        a0 = torch.cat((a0.view(a0.shape[0], -1, nu), a), dim=1)

    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()
    ts = ts.double()
    return s0.detach(), a0.detach(), sn.detach(), ts.detach()


def compute_state_actions_dim_2(rand, samples_per_dim, device_h, state_max, action_max):
    if rand:
        s0s = (
            (
                torch.rand(
                    samples_per_dim**4,
                    state_max.shape[0],
                    dtype=torch.double,
                    device=device_h,
                )
                - 0.5
            )
            * 2.0
            * state_max
        )
        actions = (
            (
                torch.rand(
                    samples_per_dim,
                    action_max.shape[0],
                    dtype=torch.double,
                    device=device_h,
                )
                - 0.5
            )
            * 2.0
            * action_max
        )
        actions = actions.view(-1)
    else:
        x, y, z, k = torch.meshgrid(
            torch.linspace(-state_max[0], state_max[0], samples_per_dim, device=device_h),
            torch.linspace(-state_max[1], state_max[1], samples_per_dim, device=device_h),
            torch.linspace(-state_max[2], state_max[2], samples_per_dim, device=device_h),
            torch.linspace(-state_max[3], state_max[3], samples_per_dim, device=device_h),
            indexing="ij",
        )
        all_t = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1), k.unsqueeze(-1)), -1)
        s0s = all_t.view(-1, 4)
        actions = torch.linspace(-action_max[0], action_max[0], samples_per_dim, device=device_h)
    return s0s, actions


def compute_state_actions(rand, samples_per_dim, device_h, state_max, action_max):
    if rand:
        s0s = (
            (
                torch.rand(
                    samples_per_dim ** state_max.shape[0],
                    state_max.shape[0],
                    dtype=torch.double,
                    device=device_h,
                )
                - 0.5
            )
            * 2.0
            * state_max
        )
        actions = (
            (
                torch.rand(
                    samples_per_dim,
                    action_max.shape[0],
                    dtype=torch.double,
                    device=device_h,
                )
                - 0.5
            )
            * 2.0
            * action_max
        )
        actions = actions.view(-1, action_max.shape[0])
    else:
        if state_max.shape[0] == 4:
            x, y, z, k = torch.meshgrid(
                torch.linspace(-state_max[0], state_max[0], samples_per_dim, device=device_h),
                torch.linspace(-state_max[1], state_max[1], samples_per_dim, device=device_h),
                torch.linspace(-state_max[2], state_max[2], samples_per_dim, device=device_h),
                torch.linspace(-state_max[3], state_max[3], samples_per_dim, device=device_h),
                indexing="ij",
            )
            all_t = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1), k.unsqueeze(-1)), -1)
            s0s = all_t.view(-1, 4)
        elif state_max.shape[0] == 2:
            x, y = torch.meshgrid(
                torch.linspace(state_max[0], state_max[0], samples_per_dim, device=device_h),
                torch.linspace(state_max[1], state_max[1], samples_per_dim, device=device_h),
                indexing="ij",
            )
            all_t = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), -1)
            s0s = all_t.view(-1, 2)
        if action_max.shape[0] == 1:
            actions = torch.linspace(-action_max[0], action_max[0], samples_per_dim, device=device_h).view(-1, 1)
        elif action_max.shape[0] == 2:
            a1, a2 = torch.meshgrid(
                torch.linspace(-action_max[0], action_max[0], samples_per_dim, device=device_h),
                torch.linspace(-action_max[1], action_max[1], samples_per_dim, device=device_h),
                indexing="ij",
            )
            a_t = torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1)
            actions = a_t.view(-1, 2)
    return s0s, actions  # pyright: ignore


def generate_irregular_data_delay_time_multi(
    train_env_task,
    env,
    delay,
    samples_per_dim=None,
    mode="grid",
    rand=False,
    action_buffer_size=5,
    encode_obs_time=False,
    reuse_state_actions_when_sampling_times=False,
):  # , time_multiplier=10): # Delay is number of timesteps dt
    if samples_per_dim is None:
        if train_env_task == "oderl-pendulum":
            samples_per_dim = 33
        elif train_env_task == "oderl-cartpole":
            samples_per_dim = 20
        elif train_env_task == "oderl-acrobot":
            samples_per_dim = 15
    time_multiplier = 10
    ACTION_HIGH = env.action_space.high[0]
    # ACTION_LOW = env.action_space.low[0]
    nu = env.action_space.shape[0]
    s0_l, a0_l, sn_l, ts_l = [], [], [], []
    action_max = torch.tensor([ACTION_HIGH] * nu)
    device_h = "cpu"
    if train_env_task == "oderl-cartpole":
        state_max = torch.tensor([5.0, 20, torch.pi, 30])  # state_max = torch.tensor([7.0, 20, torch.pi, 30])
    elif train_env_task == "oderl-pendulum":
        state_max = torch.tensor([torch.pi, 5.0])
    elif train_env_task == "oderl-acrobot":
        state_max = torch.tensor([torch.pi, torch.pi, 5.0, 5.0])
    if reuse_state_actions_when_sampling_times:
        s0s, actions = compute_state_actions(rand, samples_per_dim, device_h, state_max, action_max)  # pyright: ignore
        # pylint: disable-next=unused-variable
        for ti in range(int(samples_per_dim * time_multiplier)):  # pyright: ignore
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h)  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore
    else:
        for ti in range(int(samples_per_dim * time_multiplier)):  # pyright: ignore
            s0s, actions = compute_state_actions(
                rand, samples_per_dim, device_h, state_max, action_max  # pyright: ignore
            )
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h)  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore

    s0 = torch.cat(s0_l, dim=0)
    a0 = torch.cat(a0_l, dim=0)
    sn = torch.cat(sn_l, dim=0)
    ts = torch.cat(ts_l, dim=0)

    a = (torch.rand(a0.shape[0], action_buffer_size, nu, dtype=torch.double, device=device_h) - 0.5) * 2.0 * ACTION_HIGH
    # a = torch.zeros_like(a)
    a[:, -(delay + 1)] = a0
    a0 = a
    if encode_obs_time:
        a0 = torch.cat(
            (
                a0,
                torch.flip(torch.arange(action_buffer_size), (0,))
                .view(1, action_buffer_size, 1)
                .repeat(a0.shape[0], 1, 1),
            ),
            dim=2,
        )

    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()
    ts = ts.double()
    return s0.detach(), a0.detach(), sn.detach(), ts.detach()


def load_expert_irregular_data_delay_time_multi(
    train_env_task,
    delay,
    encode_obs_time=True,
    config=None,
):
    if config is None:
        config = dict()
    final_data = mppi_with_model_collect_data(
        "oracle",  # 'nl', 'NN', 'oracle', 'random'
        train_env_task,
        action_delay=delay,
        roll_outs=config.mppi_roll_outs,  # pyright: ignore
        time_steps=config.mppi_time_steps,  # pyright: ignore
        lambda_=config.mppi_lambda,  # pyright: ignore
        sigma=config.mppi_sigma,  # pyright: ignore
        dt=config.dt,  # pyright: ignore
        save_video=False,
        state_constraint=False,
        change_goal=False,
        encode_obs_time=encode_obs_time,
        model=None,
        uniq=None,
        log_debug=False,
        collect_samples=config.collect_expert_samples,  # pyright: ignore
        action_buffer_size=config.action_buffer_size,  # pyright: ignore
        config_in=config,
        ts_grid=config.collect_expert_ts_grid,  # pyright: ignore
        intermediate_run=False,
    )
    (s0, a0, sn, ts) = final_data

    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()
    ts = ts.double()
    return s0.detach(), a0.detach(), sn.detach(), ts.detach()


def generate_irregular_data(train_env_task, env, samples_per_dim=None, mode="grid", rand=False):
    if samples_per_dim is None:
        if train_env_task == "oderl-pendulum":
            samples_per_dim = 33
        elif train_env_task == "oderl-cartpole":
            samples_per_dim = 20
        elif train_env_task == "oderl-acrobot":
            samples_per_dim = 15

    ACTION_HIGH = env.action_space.high[0]
    # ACTION_LOW = env.action_space.low[0]
    if train_env_task == "oderl-cartpole":
        state_max = torch.tensor([5.0, 20, torch.pi, 30])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        # pylint: disable-next=unused-variable
        for ti in range(samples_per_dim):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**4, 4, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([-ACTION_HIGH, ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim, 1, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
                actions = actions.view(-1)
            else:
                x, y, z, k = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[2], state_max[2], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[3], state_max[3], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat(
                    (
                        x.unsqueeze(-1),
                        y.unsqueeze(-1),
                        z.unsqueeze(-1),
                        k.unsqueeze(-1),
                    ),
                    -1,
                )
                s0s = all_t.view(-1, 4)
                actions = torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h)
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h)  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore
    elif train_env_task == "oderl-pendulum":
        state_max = torch.tensor([torch.pi, 5.0])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        for ti in range(samples_per_dim):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**2, 2, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([-ACTION_HIGH, ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim, 1, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
                actions = actions.view(-1)
            else:
                x, y = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), -1)
                s0s = all_t.view(-1, 2)
                actions = torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h)
            s0, a0, sn, ts = env.batch_integrate_system(s0s, actions, device=device_h)  # Only support 1d actions
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore
    elif train_env_task == "oderl-acrobot":
        state_max = torch.tensor([torch.pi, torch.pi, 5.0, 5.0])
        state_min = -state_max
        device_h = "cpu"
        s0_l, a0_l, sn_l, ts_l = [], [], [], []
        for ti in tqdm(range(samples_per_dim)):  # pyright: ignore
            if rand:
                s0s = (
                    (torch.rand(samples_per_dim**4, 4, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * state_max
                )
                action_max = torch.tensor([-ACTION_HIGH, ACTION_HIGH])
                actions = (
                    (torch.rand(samples_per_dim**2, 2, dtype=torch.double, device=device_h) - 0.5)  # pyright: ignore
                    * 2.0
                    * action_max
                )
            else:
                x, y, z, k = torch.meshgrid(
                    torch.linspace(state_min[0], state_max[0], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[1], state_max[1], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[2], state_max[2], samples_per_dim, device=device_h),  # pyright: ignore
                    torch.linspace(state_min[3], state_max[3], samples_per_dim, device=device_h),  # pyright: ignore
                    indexing="ij",
                )
                all_t = torch.cat(
                    (
                        x.unsqueeze(-1),
                        y.unsqueeze(-1),
                        z.unsqueeze(-1),
                        k.unsqueeze(-1),
                    ),
                    -1,
                )
                s0s = all_t.view(-1, 4)
                if env.action_space.shape[0] == 2:
                    a1, a2 = torch.meshgrid(
                        torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h),
                        torch.linspace(-ACTION_HIGH, ACTION_HIGH, samples_per_dim, device=device_h),
                        indexing="ij",
                    )
                    a_t = torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1)
                    actions = a_t.view(-1, 2)
            s0, a0, sn, ts = env.batch_integrate_system(
                s0s, actions, device=device_h  # pyright: ignore
            )  # Stochastic sampling of time
            ts = ts.view(1).repeat(a0.shape[0]).view(-1, 1)
            # pylint: disable-next=expression-not-assigned
            s0_l.append(s0), a0_l.append(a0), sn_l.append(sn), ts_l.append(ts)  # pyright: ignore

    s0 = torch.cat(s0_l, dim=0)  # pyright: ignore
    a0 = torch.cat(a0_l, dim=0)  # pyright: ignore
    sn = torch.cat(sn_l, dim=0)  # pyright: ignore
    ts = torch.cat(ts_l, dim=0)  # pyright: ignore

    s0 = s0.double()
    a0 = a0.double()
    sn = sn.double()
    ts = ts.double()
    return s0.detach(), a0.detach(), sn.detach(), ts.detach()
