import logging
import time
from functools import partial

import imageio
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import wandb
from tqdm import tqdm

from planners.mppi_delay import MPPIDelay  # pylint: disable=import-error

from .config import dotdict, get_config, seed_all
from .overlay import create_env, setup_logger, start_virtual_display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cpu'
logger = logging.getLogger()


def get_action_with_encode_obs_time(action_buffer, action, action_delay, nu):
    action_buffer = torch.roll(action_buffer, -1, dims=0)
    action_buffer[-1, :nu] = action
    action_buffer[-1, nu:] = 0
    return action_buffer, action_buffer[-(action_delay + 1), :nu]


def get_action(action_buffer, action, action_delay):
    action_buffer = torch.roll(action_buffer, -1, dims=0)
    action_buffer[-1] = action
    return action_buffer, action_buffer[-(action_delay + 1)]


def inner_mppi_with_model_collect_data(
    seed,
    model_name,  # 'nl', 'NN', 'oracle', 'random'
    env_name,  # pylint: disable=redefined-outer-name
    action_delay,
    roll_outs=1000,
    time_steps=30,
    lambda_=1.0,
    sigma=1.0,
    dt=0.05,
    model_seed=11,
    save_video=False,
    state_constraint=False,
    change_goal=False,
    encode_obs_time=False,
    model=None,
    uniq=None,
    log_debug=False,
    episodes_per_sampler_task=10,
    action_buffer_size=4,
    config=None,  # pylint: disable=redefined-outer-name
    iter_=200,
    change_goal_flipped_iter_=False,
    ts_grid="exp",
    intermediate_run=False,
):
    if config is None:
        config = dict()
    config = dotdict(config)
    env = create_env(env_name, dt=dt, ts_grid=ts_grid, friction=config.friction)  # pyright: ignore
    ACTION_LOW = env.action_space.low[0]  # pyright: ignore
    ACTION_HIGH = env.action_space.high[0]  # pyright: ignore

    nx = env.get_obs().shape[0]  # pyright: ignore
    nu = env.action_space.shape[0]  # pyright: ignore

    dtype = torch.double
    gamma = sigma**2
    off_diagonal = 0.5 * gamma
    mppi_noise_sigma = torch.ones((nu, nu), device=device, dtype=dtype) * off_diagonal + torch.eye(
        nu, device=device, dtype=dtype
    ) * (gamma - off_diagonal)
    logger.info(mppi_noise_sigma)  # pyright: ignore
    mppi_lambda_ = 1.0

    random_action_noise = config.collect_expert_random_action_noise

    if model is None:
        # if model_name == 'nl':
        #     model = get_nl(env_name, config=config, retrain=False, delay=action_delay, model_seed=model_seed, encode_obs_time=encode_obs_time)
        # elif model_name == 'delta_t_rnn':
        #     model = w_delta_t_rnn(env_name, config=config, retrain=False, delay=action_delay)
        # elif model_name == 'node':
        #     model = get_node(env_name, config=config, retrain=False, delay=action_delay)
        if not (model_name == "oracle" or model_name == "random"):
            model.double()  # pyright: ignore

    ts_pred = torch.tensor(dt, device=device, dtype=dtype).view(1, 1).repeat(roll_outs, 1)

    if not (model_name == "oracle" or model_name == "random"):

        def dynamics(  # pyright: ignore
            state,
            perturbed_action,
            encode_obs_time=encode_obs_time,
            action_buffer_size=action_buffer_size,
            model_name=model_name,
        ):
            if encode_obs_time and model_name == "nl":
                perturbed_action = torch.cat(
                    (
                        perturbed_action,
                        torch.flip(torch.arange(action_buffer_size, device=device), (0,))
                        .view(1, action_buffer_size, 1)
                        .repeat(perturbed_action.shape[0], 1, 1),
                    ),
                    dim=2,
                )
            state_diff_pred = model(state, perturbed_action, ts_pred)  # pyright: ignore
            state_out = state + state_diff_pred
            return state_out

    elif model_name == "random":

        def dynamics(state, perturbed_action):
            pass

    elif model_name == "oracle":
        # Need to partial ts as dt !
        if env_name == "oderl-pendulum":
            from .oracle import pendulum_dynamics_dt_delay

            dynamics = pendulum_dynamics_dt_delay  # pyright: ignore
        elif env_name == "oderl-cartpole":
            from .oracle import cartpole_dynamics_dt_delay

            dynamics = cartpole_dynamics_dt_delay  # pyright: ignore
        elif env_name == "oderl-acrobot":
            from .oracle import acrobot_dynamics_dt_delay

            dynamics = acrobot_dynamics_dt_delay  # pyright: ignore
        dynamics = partial(dynamics, ts=ts_pred, delay=action_delay)  # pyright: ignore

    def running_cost(state, action):
        if state_constraint:
            reward = env.diff_obs_reward_(  # pyright: ignore
                state,
                exp_reward=False,
                state_constraint=state_constraint,  # pyright: ignore
            ) + env.diff_ac_reward_(  # pyright: ignore
                action
            )
        elif change_goal:
            global change_goal_flipped  # pylint: disable=global-variable-not-assigned
            reward = env.diff_obs_reward_(  # pyright: ignore
                state,
                exp_reward=False,
                change_goal=change_goal,  # pyright: ignore
                change_goal_flipped=change_goal_flipped,  # pyright: ignore
            ) + env.diff_ac_reward_(  # pyright: ignore
                action
            )
        else:
            reward = env.diff_obs_reward_(state, exp_reward=False) + env.diff_ac_reward_(action)  # pyright: ignore
        cost = -reward
        # if state_constraint:
        # cost = cost + (state[:,0] > 0.0).float() * - torch.nan_to_num(torch.log(1.0-state[:,0]))
        # cost = cost + torch.exp((state[:,0] > 0.0).float() * state[:,0] * 10.0)
        # cost = cost + (state[:,0] > 0.0).float() * state[:,0] * 10.0
        # cost = cost * (1.0 + 10.0 * (state[:,0] > 0.0).float() * state[:,0])
        # cost = cost - torch.nan_to_num(torch.log(1.0-state[:,0]))
        return cost

    mppi_gym = MPPIDelay(
        dynamics,  # pyright: ignore
        running_cost,
        nx,
        mppi_noise_sigma,
        num_samples=roll_outs,
        horizon=time_steps,
        device=device,  # pyright: ignore
        lambda_=mppi_lambda_,
        u_min=torch.tensor(ACTION_LOW),
        u_max=torch.tensor(ACTION_HIGH),
        u_scale=ACTION_HIGH,  # /2.0
        encode_obs_time=config.encode_obs_time,  # pyright: ignore
        dt=dt,
    )

    if save_video:
        start_virtual_display()

    videos_folder = "./logs/new_videos"
    from pathlib import Path

    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    filename = f"{videos_folder}/{env_name}_{model_name}_{uniq}.mp4"
    fps = int(1 / dt)

    def step_env(env, action, action_buffer, action_delay, obs_noise):
        at = torch.from_numpy(action).to(device)
        if encode_obs_time:
            action_buffer, at = get_action_with_encode_obs_time(action_buffer, at, action_delay=action_delay, nu=nu)
        else:
            action_buffer, at = get_action(action_buffer, at, action_delay=action_delay)

        def g(state, t):
            return at

        returns = env.integrate_system(2, g, s0=torch.tensor(env.state).to(device), return_states=True)
        state = returns[-1][-1]
        reward = returns[2][-1]
        tsn = returns[-2][-1, -1]
        if encode_obs_time:
            action_buffer[:, nu:] += tsn
            action_buffer[-1, nu:] = 0
        state += torch.randn_like(state) * obs_noise
        env.set_state_(state.cpu().numpy())
        state_out = env.get_obs()
        if env.time_step >= env.n_steps:
            logger.info(  # pyright: ignore
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive "
                "'done = True'"
            )

        env.time_step += 1
        done = True if env.time_step >= env.n_steps else False
        return state_out, reward, done, action_buffer, tsn

    def loop():
        s0 = []
        a0 = []
        sn = []
        ts = []
        ACTION_LOW = env.action_space.low[0]  # pyright: ignore
        ACTION_HIGH = env.action_space.high[0]  # pyright: ignore
        if encode_obs_time:
            action_buffer = torch.zeros((action_buffer_size, nu + 1), dtype=torch.double, device=device)
            action_buffer[:, nu:] = (torch.flip(torch.arange(4), (0,)) * dt).view(-1, 1)
        else:
            action_buffer = torch.zeros((action_buffer_size, nu), dtype=torch.double, device=device)
        it = 0
        total_reward = 0
        env.reset()
        start_time = time.perf_counter()
        mppi_gym.reset()
        while it < iter_:
            if change_goal_flipped_iter_ < it:
                # pylint: disable-next=unused-variable,redefined-outer-name
                change_goal_flipped = True  # noqa
            state = env.get_obs()  # pyright: ignore
            s0.append(state)
            command_start = time.perf_counter()
            if model_name != "random":
                action = mppi_gym.command(state, action_buffer)
                if random_action_noise is not None:
                    action += (
                        (torch.rand(nu, device=device) - 0.5) * 2.0 * env.action_space.high[0]  # pyright: ignore
                    ) * random_action_noise
                    action = action.clip(min=ACTION_LOW, max=ACTION_HIGH)
            else:
                action = torch.from_numpy(env.action_space.sample())
            # a0.append(torch.concat(action_buffer, action))
            elapsed = time.perf_counter() - command_start
            state, reward, done, action_buffer, tsn = step_env(  # pylint: disable=unused-variable
                env,
                action.detach().cpu().numpy(),
                action_buffer,
                action_delay=action_delay,
                obs_noise=config.observation_noise,  # pylint: disable=no-member
            )
            sn.append(state)
            a0.append(action_buffer)
            ts.append(tsn)
            total_reward += reward
            # print(f"action taken: {action.detach().cpu().numpy()} cost received: {-reward} |
            # state {state.flatten()} time taken: {elapsed}s | {int(it/iter_*100)}% Complete \t | iter={it}")
            if log_debug:
                # pylint: disable-next=logging-fstring-interpolation
                logger.info(  # pyright: ignore
                    f"action taken: {action.detach().cpu().numpy()} cost received: {-reward} | state {state.flatten()} "
                    f"time taken: {elapsed}s | {int(it/iter_*100)}% Complete \t | iter={it}"
                )
            if save_video:
                video.append_data(  # pyright: ignore
                    env.render(mode="rgb_array", last_act=action.detach().cpu().numpy())  # pyright: ignore
                )
            it += 1
        total_reward = total_reward.detach().cpu().item()  # pyright: ignore
        ddict = {
            "model_name": model_name,
            "env_name": env_name,
            "roll_outs": roll_outs,
            "time_steps": time_steps,
            "uniq": uniq,
            "episode_elapsed_time": time.perf_counter() - start_time,
            "dt": dt,
            "delay": action_delay,
            "planner": "mpc",
            "total_reward": total_reward,
        }
        if save_video:
            # pylint: disable-next=logging-fstring-interpolation
            logger.info(f"[Video] Watch video at : {filename}")  # pyright: ignore
        if intermediate_run:
            # pylint: disable-next=logging-fstring-interpolation
            logger.info(f"[Intermediate Result] {str(ddict)}")  # pyright: ignore
        else:
            # pylint: disable-next=logging-fstring-interpolation
            logger.info(f"[Result] {str(ddict)}")  # pyright: ignore
        s0 = torch.from_numpy(np.stack(s0))
        sn = torch.from_numpy(np.stack(sn))
        a0 = torch.stack(a0).cpu()
        ts = torch.stack(ts).cpu()
        return ddict, (s0, a0, sn, ts)

    episodes = []
    for j in range(episodes_per_sampler_task):  # pylint: disable=unused-variable
        with torch.no_grad():
            if save_video:
                with imageio.get_writer(filename, fps=fps) as video:
                    result, episode_buffer = loop()  # pylint: disable=unused-variable
                    episodes.append(episode_buffer)
            else:
                result, episode_buffer = loop()
                episodes.append(episode_buffer)
    return episodes


def mppi_with_model_collect_data(
    model_name,  # 'nl', 'NN', 'oracle', 'random'
    env_name,  # pylint: disable=redefined-outer-name
    action_delay,
    roll_outs=1000,
    time_steps=30,
    lambda_=1.0,
    sigma=1.0,
    dt=0.05,
    model_seed=11,
    save_video=False,
    state_constraint=False,
    change_goal=False,
    encode_obs_time=False,
    model=None,
    uniq=None,
    log_debug=False,
    collect_samples=1e6,
    action_buffer_size=4,
    config_in=None,
    debug_main=False,
    ts_grid="exp",
    intermediate_run=False,
):
    if config_in is None:
        config_in = dict()
    MODELS = ["nl", "oracle", "random", "delta_t_rnn", "node", "latent_ode"]
    assert model_name in MODELS
    config = dotdict(dict(config_in))  # pylint: disable=redefined-outer-name

    file_name = (
        f"replay_buffer_env-name-{env_name}_delay-{action_delay}_model-name-{model_name}"
        f"_encode-obs-time-{encode_obs_time}_action-buffer-size-{action_buffer_size}_ts-grid-{ts_grid}_"
        f"random-action-noise-{config.collect_expert_random_action_noise}_"
        f"observation-noise-{config.observation_noise}_friction-{config.friction}.pt"
    )
    if not config.collect_expert_force_generate_new_data:
        # try:
        final_data = torch.load(f"./offlinedata/{file_name}")
        return final_data
        # except FileNotFoundError as e:
        # logger.info(f'[Replay buffer not found] Unable to find replay buffer -
        # will generate a new one \t| file_name={file_name}')

    global change_goal_flipped  # pylint: disable=global-variable-undefined
    change_goal_flipped = False
    timelen = 10  # seconds
    if change_goal:
        timelen = timelen * 2.0
    iter_ = timelen / dt
    change_goal_flipped_iter_ = iter_ / 2.0

    multi_inner_mppi_with_model_collect_data = partial(
        inner_mppi_with_model_collect_data,
        model_name=model_name,  # 'nl', 'NN', 'oracle', 'random'
        env_name=env_name,
        action_delay=action_delay,
        roll_outs=roll_outs,
        time_steps=time_steps,
        lambda_=lambda_,
        sigma=sigma,
        dt=dt,
        model_seed=model_seed,
        save_video=save_video,
        state_constraint=state_constraint,
        change_goal=change_goal,
        encode_obs_time=encode_obs_time,
        model=model,
        uniq=uniq,
        log_debug=log_debug,
        episodes_per_sampler_task=config.collect_expert_episodes_per_sampler_task,  # pyright: ignore
        action_buffer_size=action_buffer_size,
        config=dict(config),
        ts_grid=ts_grid,
        iter_=iter_,  # pyright: ignore
        change_goal_flipped_iter_=change_goal_flipped_iter_,  # pyright: ignore
        intermediate_run=intermediate_run,
    )
    total_episodes_needed = int(collect_samples / iter_)
    task_inputs = [
        run_seed
        for run_seed in range(
            int(total_episodes_needed / config.collect_expert_episodes_per_sampler_task)  # pyright: ignore
        )
    ]
    episodes = []
    if not debug_main:
        pool_outer = multiprocessing.Pool(config.collect_expert_cores_per_env_sampler)
        for i, result in tqdm(  # pylint: disable=unused-variable
            enumerate(pool_outer.imap_unordered(multi_inner_mppi_with_model_collect_data, task_inputs)),
            total=len(task_inputs),
            smoothing=0,
        ):
            # print("INFO: Completed run {} of {}".format(i + 1, len(task_inputs)))
            # logger.info("INFO: Completed run {} of {}".format(i + 1, len(task_inputs)))
            episodes.extend(result)
    else:
        for i, task in tqdm(enumerate(task_inputs), total=len(task_inputs)):
            result = multi_inner_mppi_with_model_collect_data(task)
            # logger.info("INFO: Completed run {} of {}".format(i + 1, len(task_inputs)))
            episodes.extend(result)

    s0 = []
    sn = []
    a0 = []
    ts = []
    for episode in episodes:
        (es0, ea0, esn, ets) = episode
        s0.append(es0)
        sn.append(esn)
        a0.append(ea0)
        ts.append(ets)
    s0 = torch.cat(s0, dim=0)
    sn = torch.cat(sn, dim=0)
    a0 = torch.cat(a0, dim=0)
    ts = torch.cat(ts, dim=0).view(-1, 1)
    final_data = (s0, a0, sn, ts)
    torch.save(final_data, f"./offlinedata/{file_name}")
    pool_outer.close()  # pyright: ignore
    return final_data


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    defaults = get_config()
    defaults["save_video"] = False
    defaults["collect_expert_force_generate_new_data"] = True
    # defaults['ts_grid'] = 'fixed'
    # defaults['friction'] = False
    wandb.init(config=defaults, project=defaults["wandb_project"], mode="disabled")  # pyright: ignore
    config = wandb.config
    seed_all(0)

    logger = setup_logger(__file__)
    # for env_name in ['oderl-cartpole', 'oderl-acrobot', 'oderl-pendulum']:
    for env_name in ["oderl-cartpole"]:
        for delay in range(4):
            # pylint: disable-next=logging-fstring-interpolation
            logger.info(f"[Collecting data expert data] env_name={env_name} \t | delay={delay}")  # pyright: ignore
            results = mppi_with_model_collect_data(
                model_name="oracle",  # 'oracle', 'nl', 'nl', 'node'
                action_delay=delay,
                env_name=env_name,
                roll_outs=config.mppi_roll_outs,
                time_steps=config.mppi_time_steps,
                lambda_=config.mppi_lambda,
                sigma=config.mppi_sigma,
                dt=config.dt,
                collect_samples=1e6,
                uniq=0,
                debug_main=False,
                encode_obs_time=config.encode_obs_time,
                ts_grid=config.ts_grid,
                config_in=config,
                log_debug=False,
                save_video=config.save_video,
            )
    logger.info("Fin.")  # pyright: ignore
