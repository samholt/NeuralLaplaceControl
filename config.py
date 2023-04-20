import argparse
import random
from time import time

import numpy as np
import torch


def default_config():
    return dict(
        # env_name='oderl-cartpole', # must always be specified
        seed_runs=20,  # 5,
        seed_start=0,
        baselines=["nl", "oracle", "random", "delta_t_rnn", "node", "latent_ode"],
        dt=0.05,
        learning_rate=1e-4,
        collect_expert_samples=1e6,
        collect_expert_ts_grid="exp",  # 'exp',
        collect_expert_force_generate_new_data=False,
        # collect_expert_random_action_noise=.01,
        collect_expert_random_action_noise=1.0,
        collect_expert_cores_per_env_sampler=20,
        collect_expert_episodes_per_sampler_task=1,
        train_with_expert_trajectories=True,
        training_epochs=10000000,
        training_batch_size=16,  # 32,
        saved_models_path="./saved_models/",
        offline_datasets_path="./offlinedata/",
        iters_per_log=500,
        clip_grad_norm=0.1,  # 10
        normalize=True,
        normalize_time=True,
        train_dt_multiple=1,
        ts_grid="exp",  # ['fixed', 'uniform', 'exp']
        train_samples_per_dim=10,
        nl_ilt_algorithm="fourier",  # NL hyperparameters
        nl_hidden_units=128,  # 256
        nl_s_recon_terms=17,  # 33,
        # nl_encode_obs_time=True,
        node_method="euler",
        node_augment_dim=1,
        node_hidden_units=270,
        rnn_hidden_units=160,  # 64
        latent_ode_hidden_units=128,
        latent_ode_obsrv_std=0.01,
        log_folder="logs",
        weight_decay=0,
        lr_scheduler_step_size=20,
        lr_scheduler_gamma=0.1,
        use_lr_scheduler=False,
        iters_per_evaluation=1e15,  # 100e3,
        mppi_roll_outs=1000,
        mppi_time_steps=40,
        mppi_lambda=1.0,
        mppi_sigma=1.0,
        encode_obs_time=False,
        reuse_state_actions_when_sampling_times=False,
        action_buffer_size=4,
        # delay=1, # Must always be specified
        save_video=False,
        sweep_mode=False,
        end_training_after_seconds=180,
        model_seed=0,
        rand_sample=True,
        torch_deterministic=True,
        multi_process_results=True,
        retrain=False,
        force_retrain=False,
        start_from_checkpoint=True,
        print_settings=False,
        training_use_only_samples=None,
        observation_noise=0.0,
        friction=False,
        wandb_project="NeuralLaplaceControl",
    )


def parse_args(config):
    # fmt: off
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", type=str, default=config['env_name'], help="env_name")
    # Must always be specified
    parser.add_argument("--seed_runs", type=int, default=config["seed_runs"], help="seed_runs")
    parser.add_argument("--retrain", choices=("True", "False"), default=str(config["retrain"]), help="retrain")
    parser.add_argument(
        "--force_retrain", choices=("True", "False"), default=str(config["force_retrain"]), help="force_retrain"
    )
    parser.add_argument(
        "--start_from_checkpoint",
        choices=("True", "False"),
        default=str(config["start_from_checkpoint"]),
        help="start_from_checkpoint",
    )
    parser.add_argument(
        "--print_settings", choices=("True", "False"), default=str(config["print_settings"]), help="print_settings"
    )
    parser.add_argument("--seed_start", type=int, default=config["seed_start"], help="seed_start")
    # parser.add_argument("--baselines", type=str, default=config['baselines'], help="baselines")
    # parser.add_argument('-n', '--baselines', nargs="*") ?
    parser.add_argument("--dt", type=float, default=config["dt"], help="dt")
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"], help="learning_rate")
    parser.add_argument(
        "--collect_expert_samples", type=float, default=config["collect_expert_samples"], help="collect_expert_samples"
    )
    parser.add_argument("--training_epochs", type=int, default=config["training_epochs"], help="training_epochs")
    parser.add_argument(
        "--training_batch_size", type=int, default=config["training_batch_size"], help="training_batch_size"
    )
    parser.add_argument("--saved_models_path", type=str, default=config["saved_models_path"], help="saved_models_path")
    parser.add_argument(
        "--offline_datasets_path", type=str, default=config["offline_datasets_path"], help="offline_datasets_path"
    )
    parser.add_argument("--iters_per_log", type=int, default=config["iters_per_log"], help="iters_per_log")
    parser.add_argument("--clip_grad_norm", type=float, default=config["clip_grad_norm"], help="clip_grad_norm")
    parser.add_argument(
        "--collect_expert_cores_per_env_sampler",
        type=float,
        default=config["collect_expert_cores_per_env_sampler"],
        help="collect_expert_cores_per_env_sampler",
    )
    parser.add_argument(
        "--collect_expert_episodes_per_sampler_task",
        type=float,
        default=config["collect_expert_episodes_per_sampler_task"],
        help="collect_expert_episodes_per_sampler_task",
    )
    parser.add_argument("--normalize", choices=("True", "False"), default=str(config["normalize"]), help="normalize")
    parser.add_argument(
        "--normalize_time", choices=("True", "False"), default=str(config["normalize_time"]), help="normalize_time"
    )
    parser.add_argument(
        "--train_dt_multiple", type=float, default=config["train_dt_multiple"], help="train_dt_multiple"
    )
    parser.add_argument(
        "--collect_expert_random_action_noise",
        type=float,
        default=config["collect_expert_random_action_noise"],
        help="collect_expert_random_action_noise",
    )
    parser.add_argument("--ts_grid", type=str, default=config["ts_grid"], help="ts_grid")
    parser.add_argument(
        "--train_samples_per_dim", type=int, default=config["train_samples_per_dim"], help="train_samples_per_dim"
    )
    parser.add_argument("--nl_ilt_algorithm", type=str, default=config["nl_ilt_algorithm"], help="nl_ilt_algorithm")
    parser.add_argument("--nl_hidden_units", type=int, default=config["nl_hidden_units"], help="nl_hidden_units")
    parser.add_argument("--nl_s_recon_terms", type=int, default=config["nl_s_recon_terms"], help="nl_s_recon_terms")
    # parser.add_argument(
    # "--nl_encode_obs_time", choices=('True','False'), default=str(config['nl_encode_obs_time']),
    # help="nl_encode_obs_time")
    parser.add_argument("--node_method", type=str, default=config["node_method"], help="node_method")
    parser.add_argument("--node_augment_dim", type=int, default=config["node_augment_dim"], help="node_augment_dim")
    parser.add_argument("--node_hidden_units", type=int, default=config["node_hidden_units"], help="node_hidden_units")
    parser.add_argument("--rnn_hidden_units", type=int, default=config["rnn_hidden_units"], help="rnn_hidden_units")
    parser.add_argument(
        "--latent_ode_hidden_units", type=int, default=config["latent_ode_hidden_units"], help="latent_ode_hidden_units"
    )
    parser.add_argument(
        "--lr_scheduler_step_size", type=int, default=config["lr_scheduler_step_size"], help="lr_scheduler_step_size"
    )
    parser.add_argument(
        "--lr_scheduler_gamma", type=float, default=config["lr_scheduler_gamma"], help="lr_scheduler_gamma"
    )
    parser.add_argument(
        "--latent_ode_obsrv_std", type=float, default=config["latent_ode_obsrv_std"], help="latent_ode_obsrv_std"
    )
    parser.add_argument("--weight_decay", type=float, default=config["weight_decay"], help="weight_decay")
    parser.add_argument("--log_folder", type=str, default=config["log_folder"], help="log_folder")
    parser.add_argument(
        "--iters_per_evaluation", type=float, default=config["iters_per_evaluation"], help="iters_per_evaluation"
    )
    parser.add_argument("--mppi_roll_outs", type=int, default=config["mppi_roll_outs"], help="mppi_roll_outs")
    parser.add_argument("--mppi_time_steps", type=int, default=config["mppi_time_steps"], help="mppi_time_steps")
    parser.add_argument("--mppi_lambda", type=float, default=config["mppi_lambda"], help="mppi_lambda")
    parser.add_argument("--mppi_sigma", type=float, default=config["mppi_sigma"], help="mppi_sigma")
    parser.add_argument(
        "--encode_obs_time", choices=("True", "False"), default=str(config["encode_obs_time"]), help="encode_obs_time"
    )
    parser.add_argument(
        "--reuse_state_actions_when_sampling_times",
        choices=("True", "False"),
        default=str(config["reuse_state_actions_when_sampling_times"]),
        help="reuse_state_actions_when_sampling_times",
    )
    parser.add_argument(
        "--action_buffer_size", type=int, default=config["action_buffer_size"], help="action_buffer_size"
    )
    # parser.add_argument("--delay", type=int, default=config['delay'], help="delay") # Must always be specified
    parser.add_argument("--model_seed", type=int, default=config["model_seed"], help="model_seed")
    parser.add_argument("--save_video", choices=("True", "False"), default=str(config["save_video"]), help="save_video")
    parser.add_argument("--sweep_mode", choices=("True", "False"), default=str(config["sweep_mode"]), help="sweep_mode")
    parser.add_argument(
        "--rand_sample", choices=("True", "False"), default=str(config["rand_sample"]), help="rand_sample"
    )
    parser.add_argument(
        "--collect_expert_force_generate_new_data",
        choices=("True", "False"),
        default=str(config["collect_expert_force_generate_new_data"]),
        help="collect_expert_force_generate_new_data",
    )
    parser.add_argument(
        "--train_with_expert_trajectories",
        choices=("True", "False"),
        default=str(config["train_with_expert_trajectories"]),
        help="train_with_expert_trajectories",
    )
    parser.add_argument(
        "--end_training_after_seconds",
        type=float,
        default=config["end_training_after_seconds"],
        help="end_training_after_seconds",
    )
    parser.add_argument(
        "--torch_deterministic",
        choices=("True", "False"),
        default=str(config["torch_deterministic"]),
        help="torch_deterministic",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        choices=("True", "False"),
        default=str(config["use_lr_scheduler"]),
        help="use_lr_scheduler",
    )
    parser.add_argument(
        "--multi_process_results",
        choices=("True", "False"),
        default=str(config["multi_process_results"]),
        help="multi_process_results",
    )
    parser.add_argument(
        "--observation_noise", type=float, default=config["observation_noise"], help="observation_noise"
    )
    parser.add_argument("--friction", choices=("True", "False"), default=str(config["friction"]), help="friction")
    args = parser.parse_args()
    ddict = vars(args)
    ddict['normalize'] = ddict['normalize'] == 'True'
    ddict['normalize_time'] = ddict['normalize_time'] == 'True'
    # ddict['nl_encode_obs_time'] = ddict['nl_encode_obs_time'] == 'True'
    ddict['encode_obs_time'] = ddict['encode_obs_time'] == 'True'
    ddict['reuse_state_actions_when_sampling_times'] = ddict['reuse_state_actions_when_sampling_times'] == 'True'
    ddict['save_video'] = ddict['save_video'] == 'True'
    ddict['sweep_mode'] = ddict['sweep_mode'] == 'True'
    ddict['rand_sample'] = ddict['rand_sample'] == 'True'
    ddict['collect_expert_force_generate_new_data'] = ddict['collect_expert_force_generate_new_data'] == 'True'
    ddict['train_with_expert_trajectories'] = ddict['train_with_expert_trajectories'] == 'True'
    ddict['torch_deterministic'] = ddict['torch_deterministic'] == 'True'
    ddict['use_lr_scheduler'] = ddict['use_lr_scheduler'] == 'True'
    ddict['multi_process_results'] = ddict['multi_process_results'] == 'True'
    ddict['retrain'] = ddict['retrain'] == 'True'
    ddict['force_retrain'] = ddict['force_retrain'] == 'True'
    ddict['start_from_checkpoint'] = ddict['start_from_checkpoint'] == 'True'
    ddict['print_settings'] = ddict['print_settings'] == 'True'
    ddict['friction'] = ddict['friction'] == 'True'
    # fmt: on
    return ddict


def get_config():
    defaults = default_config()
    args = parse_args(defaults)
    defaults.update(args)
    return defaults


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # pyright: ignore
    __delattr__ = dict.__delitem__  # pyright: ignore


def default_config_dd():
    d_c = default_config()
    return dotdict(d_c)


def CME_reconstruction_terms():
    return np.array(
        [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            101,
            111,
            121,
            131,
            141,
            151,
            161,
            171,
            181,
            191,
            201,
            211,
            216,
            221,
            231,
            241,
            251,
            261,
            271,
            281,
            291,
            301,
            311,
            321,
            331,
            341,
            351,
            361,
            371,
            381,
            391,
            396,
            401,
            421,
            441,
            461,
            481,
            501,
            521,
            541,
            561,
            581,
            601,
            621,
            641,
            661,
            681,
            701,
            721,
            741,
            761,
            781,
            801,
            821,
            841,
            861,
            881,
            901,
            921,
            941,
            961,
            981,
            1001,
        ]
    )


def seed_all(seed=None):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config. If there is no seed or it is None, a time-based
    seed is used instead and is written to config.
    """
    # Default uses current time in milliseconds, modulo 1e9
    if seed is None:
        seed = round(time() * 1000) % int(1e9)

    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
