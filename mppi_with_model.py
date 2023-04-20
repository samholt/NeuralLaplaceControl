import time
from functools import partial

import imageio
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import wandb
from torch.multiprocessing import get_logger
from tqdm import tqdm

from planners.mppi_delay import MPPIDelay  # pylint: disable=import-error

from .config import dotdict, get_config, seed_all
from .overlay import create_env, setup_logger, start_virtual_display

# import logging

# logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cpu'

logger = get_logger()


def get_action(action_buffer, action, action_delay):
    action_buffer = torch.roll(action_buffer, -1, dims=0)
    action_buffer[-1] = action
    return action_buffer, action_buffer[-(action_delay + 1)]


def mppi_with_model_evaluate_single_step(
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
    action_buffer_size=4,
    config=None,  # pylint: disable=redefined-outer-name
    intermediate_run=False,
    seed=None,  # pylint: disable=redefined-outer-name
):
    if config is None:
        config = dict()
    MODELS = ["nl", "oracle", "random", "delta_t_rnn", "rnn", "node", "latent_ode"]
    assert model_name in MODELS
    env = create_env(env_name, dt=dt, friction=config.friction)  # pyright: ignore
    ACTION_LOW = env.action_space.low[0]  # pyright: ignore
    ACTION_HIGH = env.action_space.high[0]  # pyright: ignore
    delay = action_delay  # pylint: disable=redefined-outer-name
    config.multi_process_results = True  # pyright: ignore

    nx = env.get_obs().shape[0]  # pyright: ignore
    nu = env.action_space.shape[0]  # pyright: ignore

    dtype = torch.double
    gamma = sigma**2
    off_diagonal = 0.5 * gamma
    mppi_noise_sigma = torch.ones((nu, nu), device=device, dtype=dtype) * off_diagonal + torch.eye(
        nu, device=device, dtype=dtype
    ) * (gamma - off_diagonal)
    # logger.info(mppi_noise_sigma)
    mppi_lambda_ = 1.0

    ts_pred = torch.tensor(dt, device=device, dtype=dtype).view(1, 1).repeat(roll_outs, 1)

    if not (model_name == "oracle" or model_name == "random"):
        if model is None:
            from .train_utils import train_model

            # pylint: disable-next=unused-variable
            model, results = train_model(
                model_name,
                env_name,
                config=config,
                delay=action_delay,
                wandb=None,
                model_seed=config.model_seed,  # pyright: ignore
                retrain=False,
                start_from_checkpoint=True,
                force_retrain=False,
                print_settings=False,
                evaluate_model_when_trained=False,
            )
            # if model_name == 'nl':
            #     model = get_nl(env_name, config=config, retrain=False, delay=action_delay,\
            #     model_seed=model_seed, encode_obs_time=encode_obs_time)
            # elif model_name == 'delta_t_rnn':
            #     model = w_delta_t_rnn(env_name, config=config, retrain=False, delay=action_delay)
            # elif model_name == 'node':
            #     model = get_node(env_name, config=config, retrain=False, delay=action_delay)
            model.double()

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
            state_diff_pred = model(state, perturbed_action, ts_pred)
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
        dynamics = partial(dynamics, ts=ts_pred, delay=action_delay, friction=config.friction)  # pyright: ignore

    def running_cost(state, action):
        if state_constraint:
            reward = env.diff_obs_reward_(  # pyright: ignore
                state, exp_reward=False, state_constraint=state_constraint  # pyright: ignore
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

    # pylint: disable-next=unused-variable
    def retrain_dynamics(data):
        pass

    # downward_start = True
    # j = 0
    videos_folder = "./logs/new_videos"
    from pathlib import Path

    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    filename = f"{videos_folder}/{env_name}_{model_name}_{uniq}.mp4"
    fps = int(1 / dt)
    env.reset()
    # env.set_state_(np.array([0.0,0.0,0.0,0.0]))
    state = env.get_obs()  # pyright: ignore  # pylint: disable=unused-variable
    if "pendulum" in env_name:  # Start pendulum env in downward position
        env.state = np.array([np.pi, 1])  # pyright: ignore
    # if 'cartpole' in env_name: # Start cartpole env in upward position
    #     env.state = np.array([0.0, 0.0, 0.0, 0.0])

    def step_env(env, action, action_buffer, action_delay, obs_noise):
        at = torch.from_numpy(action).to(device)
        action_buffer, at = get_action(action_buffer, at, action_delay=action_delay)

        def g(state, t):
            return at

        returns = env.integrate_system(2, g, s0=torch.tensor(env.state).to(device), return_states=True)
        state = returns[-1][-1]
        reward = returns[2][-1]
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
        return state_out, reward, done, action_buffer

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
    )
    global change_goal_flipped  # pylint: disable=global-variable-undefined
    change_goal_flipped = False
    # retrain_after_iter = 50
    # iter_=200
    timelen = 10  # seconds
    if change_goal:
        timelen = timelen * 2.0
    iter_ = timelen / dt
    change_goal_flipped_iter_ = iter_ / 2.0

    if save_video:
        start_virtual_display()

    def loop():
        action_buffer = torch.zeros((action_buffer_size, nu), dtype=torch.double, device=device)
        it = 0
        total_reward = 0
        # start_time = time.perf_counter()
        episode_elapsed_time = 0
        while it < iter_:
            if change_goal_flipped_iter_ < it:
                # pylint: disable-next=unused-variable,redefined-outer-name
                change_goal_flipped = True  # noqa
            state = env.get_obs()  # pyright: ignore
            command_start = time.perf_counter()
            if model_name != "random":
                t0 = time.perf_counter()
                action = mppi_gym.command(state, action_buffer)
                episode_elapsed_time += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                action = torch.from_numpy(env.action_space.sample())
                episode_elapsed_time += time.perf_counter() - t0
            elapsed = time.perf_counter() - command_start
            state, reward, done, action_buffer = step_env(  # pylint: disable=unused-variable
                env,
                action.detach().cpu().numpy(),
                action_buffer,
                action_delay=action_delay,
                obs_noise=config.observation_noise,  # pyright: ignore
            )
            total_reward += reward
            if not config.multi_process_results:  # pyright: ignore
                # pylint: disable-next=logging-fstring-interpolation
                logger.info(  # pyright: ignore
                    f"[{env_name}\t{model_name}\td={delay}|time_steps={time_steps}__dt={dt}] action taken: "
                    f"{action.detach().cpu().numpy()} cost received: {-reward} | state {state.flatten()} "
                    f"time taken: {elapsed}s | {int(it/iter_*100)}% Complete \t | iter={it}"
                )
            # print(f"action taken: {action.detach().cpu().numpy()} cost received: {-reward} | state {state.flatten()}
            # time taken: {elapsed}s | {int(it/iter_*100)}% Complete \t | iter={it}")
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
            "episode_elapsed_time": episode_elapsed_time,
            "episode_elapsed_time_per_it": episode_elapsed_time / it,
            "dt": dt,
            "delay": action_delay,
            "planner": "mpc",
            "total_reward_raw": total_reward,
            "total_reward": total_reward * (200.0 / iter_),
        }
        if not config.multi_process_results:  # pyright: ignore
            if save_video:
                # pylint: disable-next=logging-fstring-interpolation
                logger.info(  # pyright: ignore
                    f"[{env_name}\t{model_name}\td={delay}][Video] Watch video at : {filename}"
                )
            if intermediate_run:
                # pylint: disable-next=logging-fstring-interpolation
                logger.info(  # pyright: ignore
                    f"[{env_name}\t{model_name}\td={delay}][Intermediate Result] {str(ddict)}"
                )
            else:
                # pylint: disable-next=logging-fstring-interpolation
                logger.info(f"[{env_name}\t{model_name}\td={delay}][Result] {str(ddict)}")  # pyright: ignore
        return ddict

    with torch.no_grad():
        if save_video:
            with imageio.get_writer(filename, fps=fps) as video:
                result = loop()  # pylint: disable=redefined-outer-name
        else:
            result = loop()
    return result


def seed_wrapper_mppi_with_model_evaluate_single_step(
    seed,  # pylint: disable=redefined-outer-name
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
    action_buffer_size=4,
    config=None,  # pylint: disable=redefined-outer-name
    intermediate_run=False,
):
    if config is None:
        config = dict()
    seed_all(seed)
    config = dotdict(config)
    results = mppi_with_model_evaluate_single_step(
        model_name=model_name,  # 'oracle', 'nl', 'nl', 'node'
        action_delay=action_delay,
        env_name=env_name,
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
        action_buffer_size=action_buffer_size,
        config=config,
        intermediate_run=intermediate_run,
    )
    return results


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    defaults = get_config()
    defaults["save_video"] = True
    defaults["collect_expert_cores_per_env_sampler"] = 19
    debug_main = True
    # defaults['friction'] = False
    wandb.init(config=defaults, project=defaults["wandb_project"], mode="disabled")  # pyright: ignore
    config = wandb.config
    logger = setup_logger(__file__)

    if not debug_main:
        pool_outer = multiprocessing.Pool(config.collect_expert_cores_per_env_sampler)
    # for env_name in ['oderl-cartpole', 'oderl-acrobot', 'oderl-pendulum']:
    for env_name in ["oderl-cartpole"]:
        for delay in range(4):
            total_rewards = []
            seeds = range(config.seed_start, config.seed_runs + config.seed_start)
            multi_seed_wrapper_mppi_with_model_evaluate_single_step = partial(
                seed_wrapper_mppi_with_model_evaluate_single_step,
                # model_name='oracle',  # 'oracle', 'nl', 'nl', 'node'
                model_name="nl",  # 'oracle', 'nl', 'nl', 'node'
                action_delay=delay,
                env_name=env_name,
                roll_outs=config.mppi_roll_outs,
                time_steps=config.mppi_time_steps,
                lambda_=config.mppi_lambda,
                sigma=config.mppi_sigma,
                dt=config.dt,
                uniq=0,
                encode_obs_time=config.encode_obs_time,
                config=dict(config),
                save_video=config.save_video,
            )
            if debug_main:
                for i, seed in tqdm(enumerate(seeds), total=len(seeds)):
                    result = multi_seed_wrapper_mppi_with_model_evaluate_single_step(seed)
                    total_rewards.append(result["total_reward"])
            else:
                for i, result in tqdm(
                    enumerate(
                        # pylint: disable-next=used-before-assignment
                        pool_outer.imap_unordered(  # pyright: ignore
                            multi_seed_wrapper_mppi_with_model_evaluate_single_step,
                            seeds,
                        )
                    ),
                    total=len(seeds),
                ):
                    total_rewards.append(result["total_reward"])
            # pylint: disable-next=logging-fstring-interpolation
            logger.info(  # pyright: ignore
                f"[Total average reward] env_name={env_name}\t\tdelay={delay} \t| "
                f"{np.mean(total_rewards)} +/- {np.std(total_rewards)}"
            )
    if not debug_main:
        pool_outer.close()  # pyright: ignore
    logger.info("Fin.")  # pyright: ignore
    # wandb.log({'total_reward': results['total_reward'], 'episode_elapsed_time': results['episode_elapsed_time']})
